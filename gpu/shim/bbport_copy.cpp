// SPDX-License-Identifier: GPL-2.0-or-later
#include "bbport_copy.h"
#include "bbport_threads.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
#endif

#include "bbport_toggles.h"
#include "common/thread.h"

namespace BbCopy {
namespace {

thread_local bool in_copy_thread = false;

class Pool {
public:
    Pool() {
        // Copies are on the critical path (fences wait for them): normal priority, a quarter
        // of the hardware threads available to the process (Steam Deck 2).
        unsigned count = std::clamp(BbThreads::Available() / 4, 1u, 4u);
        if (const char* env = std::getenv("BB_COPY_THREADS")) {
            count = static_cast<unsigned>(std::clamp(std::atoi(env), 0, 16));
        }
        free_batches.reserve(16);
        for (unsigned i = 0; i < count; ++i) {
            threads.emplace_back([this, i] { Loop(i); });
        }
    }
    ~Pool() {
        {
            std::scoped_lock lk{mutex};
            stop = true;
        }
        cv.notify_all();
        for (auto& thread : threads) {
            thread.join();
        }
    }

    bool Enabled() const {
        return !threads.empty();
    }

    u64 IssuedSeq() {
        std::scoped_lock lk{mutex};
        return issued_seq;
    }

    u64 Async(std::function<void()> task) {
        u64 seq;
        {
            std::scoped_lock lk{mutex};
            seq = ++issued_seq;
            ++pending;
            inflight_done.push_back(0);
            async_tasks.push_back({std::move(task), {}, seq});
        }
        cv.notify_one();
        return seq;
    }

    u64 SubmitBatch(std::vector<Item>& caller_items) {
        std::vector<Item> batch_items;
        {
            std::scoped_lock lk{mutex};
            const u64 seq = ++issued_seq;
            ++pending;
            inflight_done.push_back(0);
            if (!free_batches.empty()) {
                batch_items = std::move(free_batches.back());
                free_batches.pop_back();
            }
            caller_items.swap(batch_items);
            if (caller_items.capacity() < 256) {
                caller_items.reserve(256);
            }
            async_tasks.push_back({{}, std::move(batch_items), seq});
            cv.notify_one();
            return seq;
        }
    }

    bool IsComplete(u64 target_seq) const {
        return completed_seq_atomic.load(std::memory_order_acquire) >= target_seq;
    }

    void AfterSeq(u64 target_seq, std::function<void()> callback) {
        {
            std::unique_lock lk{mutex};
            if (completed_seq < target_seq || !waiters.empty() || draining) {
                // Runs once every task with seq <= target_seq is done, in order with earlier ones.
                waiters.push_back({target_seq, std::move(callback)});
                return;
            }
        }
        callback();
    }

    void WaitUntil(u64 target_seq) {
        if (IsComplete(target_seq)) {
            return;
        }
        u32 spins = 0;
        while (!IsComplete(target_seq)) {
            Task task;
            bool popped = false;
            {
                std::scoped_lock lk{mutex};
                if (completed_seq >= target_seq) {
                    return;
                }
                if (!async_tasks.empty() && async_tasks.front().seq <= target_seq) {
                    task = std::move(async_tasks.front());
                    async_tasks.pop_front();
                    popped = true;
                }
            }
            if (popped) {
                RunAsync(task);
                spins = 0;
            } else if (spins < 64) {
                ++spins;
#if defined(__x86_64__) || defined(_M_X64)
                __builtin_ia32_pause();
#endif
            } else {
                std::this_thread::yield();
            }
        }
    }

    void WaitAll() {
        // The waiting thread helps instead of only spinning.
        while (true) {
            Task task;
            bool popped = false;
            {
                std::scoped_lock lk{mutex};
                if (pending == 0 && waiters.empty() && !draining) {
                    return;
                }
                if (!async_tasks.empty()) {
                    task = std::move(async_tasks.front());
                    async_tasks.pop_front();
                    popped = true;
                }
            }
            if (popped) {
                RunAsync(task);
            } else {
                std::this_thread::yield();
            }
        }
    }

    void Run(std::size_t count, const std::function<void(std::size_t)>& task) {
        auto job = std::make_shared<Job>();
        job->task = &task;
        job->count = count;
        {
            std::scoped_lock lk{mutex};
            current = job;
            ++generation;
        }
        cv.notify_all();
        Work(*job);
        // Workers only call the task for indices below count, all done before this returns.
        while (job->done.load(std::memory_order_acquire) < count) {
            std::this_thread::yield();
        }
        std::scoped_lock lk{mutex};
        if (current == job) {
            current.reset();
        }
    }

private:
    using u64 = unsigned long long;
    struct Task {
        std::function<void()> run;
        std::vector<Item> items;
        u64 seq{};
    };
    struct Waiter {
        u64 target_seq{};
        std::function<void()> callback;
    };
    struct Job {
        const std::function<void(std::size_t)>* task{};
        std::size_t count{};
        std::atomic<std::size_t> next{0};
        std::atomic<std::size_t> done{0};
    };

    void RunAsync(Task& task) {
        const bool was = in_copy_thread;
        in_copy_thread = true;
        if (task.run) {
            task.run();
        } else {
            for (const auto& item : task.items) {
                item.run(item);
            }
            task.items.clear();
        }
        in_copy_thread = was;
        Complete(task.seq, std::move(task.items));
    }

    /// Retires a task and runs waiters whose target_seq has completed, oldest first. One
    /// thread drains at a time so the callbacks keep their order.
    void Complete(u64 seq, std::vector<Item>&& recycled) {
        std::unique_lock lk{mutex};
        if (recycled.capacity() > 0 && free_batches.size() < 32) {
            recycled.clear();
            free_batches.push_back(std::move(recycled));
        }
        --pending;
        inflight_done[seq - completed_seq - 1] = 1;
        while (!inflight_done.empty() && inflight_done.front()) {
            inflight_done.pop_front();
            ++completed_seq;
        }
        completed_seq_atomic.store(completed_seq, std::memory_order_release);
        if (draining) {
            return;
        }
        draining = true;
        while (true) {
            std::vector<std::function<void()>> ready;
            while (!waiters.empty() && waiters.front().target_seq <= completed_seq) {
                ready.push_back(std::move(waiters.front().callback));
                waiters.pop_front();
            }
            if (ready.empty()) {
                break;
            }
            lk.unlock();
            for (auto& callback : ready) {
                callback();
            }
            lk.lock();
        }
        draining = false;
    }

    static void Work(Job& job) {
        const bool was = in_copy_thread;
        in_copy_thread = true;
        for (std::size_t i; (i = job.next.fetch_add(1, std::memory_order_relaxed)) < job.count;) {
            (*job.task)(i);
            job.done.fetch_add(1, std::memory_order_release);
        }
        in_copy_thread = was;
    }

    void Loop(unsigned index) {
        Common::SetCurrentThreadName(("bb:Copy" + std::to_string(index)).c_str());
        u64 seen = 0;
        std::unique_lock lk{mutex};
        while (true) {
            cv.wait(lk, [&] {
                return stop || !async_tasks.empty() || (generation != seen && current);
            });
            if (stop) {
                return;
            }
            if (!async_tasks.empty()) {
                Task task = std::move(async_tasks.front());
                async_tasks.pop_front();
                lk.unlock();
                RunAsync(task);
                lk.lock();
                continue;
            }
            seen = generation;
            const auto job = current;
            lk.unlock();
            Work(*job);
            lk.lock();
        }
    }

    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable cv;
    std::shared_ptr<Job> current;
    u64 generation = 0;
    bool stop = false;
    std::deque<Task> async_tasks;
    std::deque<unsigned char> inflight_done;
    std::deque<Waiter> waiters;
    std::vector<std::vector<Item>> free_batches;
    u64 issued_seq = 0;
    u64 completed_seq = 0;
    alignas(64) std::atomic<u64> completed_seq_atomic{0};
    std::size_t pending = 0;
    bool draining = false;
};

Pool& GetPool() {
    static Pool pool;
    return pool;
}

struct Batch {
    std::vector<Item> items;
    unsigned long long bytes = 0;
};
thread_local Batch batch;
thread_local unsigned long long last_submitted_seq = 0;

} // namespace

bool Enabled() {
    return GetPool().Enabled() && !BbToggle::Disabled(BbToggle::ParallelCopies);
}

void Async(std::function<void()> task) {
    if (!Enabled()) {
        task();
        return;
    }
    last_submitted_seq = GetPool().Async(std::move(task));
}

unsigned long long FlushBatch() {
    if (batch.items.empty()) {
        return last_submitted_seq;
    }
    if (!Enabled()) {
        for (const auto& item : batch.items) {
            item.run(item);
        }
        batch.items.clear();
        batch.bytes = 0;
        return last_submitted_seq;
    }
    batch.bytes = 0;
    last_submitted_seq = GetPool().SubmitBatch(batch.items);
    return last_submitted_seq;
}

bool IsComplete(unsigned long long seq) {
    return seq == 0 || GetPool().IsComplete(seq);
}

void WaitUntil(unsigned long long seq) {
    if (seq == 0) {
        return;
    }
    GetPool().WaitUntil(seq);
}

void AfterSeq(unsigned long long seq, std::function<void()> callback) {
    if (seq == 0 || !Enabled()) {
        callback();
        return;
    }
    GetPool().AfterSeq(seq, std::move(callback));
}

void QueueCopy(const Item& item) {
    if (!Enabled()) {
        item.run(item);
        return;
    }
    // Smaller batches (32 items / 64 KiB, done before the next fence) measured slower.
    batch.items.push_back(item);
    batch.bytes += item.size;
    if (batch.bytes >= 512 * 1024 || batch.items.size() >= 256) {
        FlushBatch();
    }
}

void AfterCopies(std::function<void()> callback) {
    FlushBatch();
    if (!Enabled()) {
        callback();
        return;
    }
    auto& pool = GetPool();
    pool.AfterSeq(pool.IssuedSeq(), std::move(callback));
}

void WaitAsync() {
    // The caller's own batch runs here: handing it over only to wait for it costs a wakeup.
    if (!batch.items.empty()) {
        for (const auto& item : batch.items) {
            item.run(item);
        }
        batch.items.clear();
        batch.bytes = 0;
    }
    GetPool().WaitAll();
}

void ParallelFor(std::size_t count, const std::function<void(std::size_t)>& task) {
    if (count <= 1 || in_copy_thread || !Enabled()) {
        for (std::size_t i = 0; i < count; ++i) {
            task(i);
        }
        return;
    }
    GetPool().Run(count, task);
}

} // namespace BbCopy
