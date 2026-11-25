// SPDX-FileCopyrightText: Copyright DXVK Project
// SPDX-License-Identifier: Zlib

#pragma once

#include <array>
#include <vector>
#include "common/types.h"

namespace Vulkan {

enum class Access : u32 {
    None = 0,
    Read = 1,
    Write = 2,
};

struct AddressRange {
    u64 resource{};
    u64 range_start{};
    u64 range_end{};

    bool Contains(const AddressRange& other) const {
        return resource == other.resource && range_start <= other.range_start &&
               range_end >= other.range_end;
    }

    bool Overlaps(const AddressRange& other) const {
        return resource == other.resource && range_end >= other.range_start &&
               range_start <= other.range_end;
    }

    bool operator<(const AddressRange& other) const {
        return (resource < other.resource) ||
               (resource == other.resource && range_start < other.range_start);
    }
};

/**
 * Provides coalesced sorted-interval tracking for read and written resource ranges.
 */
class BarrierTracker {
public:
    explicit BarrierTracker();
    ~BarrierTracker() = default;

    /// Checks whether there is a pending access of a given type
    bool FindRange(const AddressRange& range, Access access_type) const;

    /// Inserts address range for a given access type
    void InsertRange(const AddressRange& range, Access access_type);

    /// Clears the entire structure
    void Clear();

    /// Checks whether any resources are dirty
    bool Empty() const {
        return active_buckets[0] == 0 && active_buckets[1] == 0;
    }

private:
    struct Interval {
        u64 start{};
        u64 end{};
    };

    struct Bucket {
        u64 resource{};
        u64 min_start{};
        u64 max_end{};
        std::vector<Interval> intervals;
    };

    std::array<std::vector<Bucket>, 2> buckets{}; // 0 = Read, 1 = Write
    std::array<u32, 2> active_buckets{};
};

} // namespace Vulkan
