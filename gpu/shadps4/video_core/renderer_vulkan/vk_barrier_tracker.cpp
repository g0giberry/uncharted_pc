// SPDX-FileCopyrightText: Copyright DXVK Project
// SPDX-License-Identifier: Zlib

#include <algorithm>
#include "video_core/renderer_vulkan/vk_barrier_tracker.h"

namespace Vulkan {

BarrierTracker::BarrierTracker() {
    buckets[0].reserve(8);
    buckets[1].reserve(8);
}

bool BarrierTracker::FindRange(const AddressRange& range, Access access_type) const {
    if (range.range_start > range.range_end) {
        return false;
    }

    const u32 idx = access_type == Access::Write ? 1u : 0u;
    const auto& list = buckets[idx];
    const u32 count = active_buckets[idx];
    const Bucket* found = nullptr;
    for (u32 i = 0; i < count; ++i) {
        if (list[i].resource == range.resource) {
            found = &list[i];
            break;
        }
    }
    if (!found) {
        return false;
    }

    if (range.range_end < found->min_start || range.range_start > found->max_end) {
        return false;
    }
    if (found->intervals.size() == 1) {
        return true;
    }

    const auto it = std::lower_bound(
        found->intervals.begin(), found->intervals.end(), range.range_start,
        [](const Interval& iv, u64 start) { return iv.end < start; });
    return it != found->intervals.end() && it->start <= range.range_end;
}

void BarrierTracker::InsertRange(const AddressRange& range, Access access_type) {
    if (range.range_start > range.range_end) {
        return;
    }

    const u32 idx = access_type == Access::Write ? 1u : 0u;
    auto& list = buckets[idx];
    const u32 count = active_buckets[idx];
    Bucket* found = nullptr;
    for (u32 i = 0; i < count; ++i) {
        if (list[i].resource == range.resource) {
            found = &list[i];
            break;
        }
    }
    if (!found) {
        if (count == list.size()) {
            list.emplace_back();
            list.back().intervals.reserve(16);
        }
        auto& b = list[active_buckets[idx]++];
        b.resource = range.resource;
        b.min_start = range.range_start;
        b.max_end = range.range_end;
        b.intervals.push_back({range.range_start, range.range_end});
        return;
    }

    // Monotonic / adjacent-tail fast path.
    auto& back = found->intervals.back();
    if (range.range_start >= back.start) {
        if (range.range_end <= back.end) {
            return;
        }
        if (back.end == ~u64{0} || range.range_start <= back.end + 1) {
            back.end = range.range_end;
            found->max_end = range.range_end;
            return;
        }
        found->intervals.push_back({range.range_start, range.range_end});
        found->max_end = range.range_end;
        return;
    }

    const u64 search_start = range.range_start > 0 ? range.range_start - 1 : 0;
    const auto it = std::lower_bound(
        found->intervals.begin(), found->intervals.end(), search_start,
        [](const Interval& iv, u64 v) { return iv.end < v; });
    const bool can_merge =
        it != found->intervals.end() &&
        (range.range_end == ~u64{0} || it->start <= range.range_end + 1);
    if (!can_merge) {
        found->intervals.insert(it, {range.range_start, range.range_end});
    } else {
        if (it->start <= range.range_start && it->end >= range.range_end) {
            return;
        }
        const u64 merged_start = std::min(it->start, range.range_start);
        u64 merged_end = std::max(it->end, range.range_end);
        auto merge_end = it + 1;
        while (merge_end != found->intervals.end() &&
               (merged_end == ~u64{0} || merge_end->start <= merged_end + 1)) {
            merged_end = std::max(merged_end, merge_end->end);
            ++merge_end;
        }
        it->start = merged_start;
        it->end = merged_end;
        if (merge_end != it + 1) {
            found->intervals.erase(it + 1, merge_end);
        }
    }
    found->min_start = found->intervals.front().start;
    found->max_end = found->intervals.back().end;
}

void BarrierTracker::Clear() {
    for (u32 a = 0; a < 2; ++a) {
        for (u32 i = 0; i < active_buckets[a]; ++i) {
            buckets[a][i].intervals.clear();
        }
        active_buckets[a] = 0;
    }
}

} // namespace Vulkan
