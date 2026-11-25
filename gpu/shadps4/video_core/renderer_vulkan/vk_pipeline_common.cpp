// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <span>

#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>

#include "shader_recompiler/resource.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_common.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

Pipeline::Pipeline(const Instance& instance_, Scheduler& scheduler_, DescriptorHeap& desc_heap_,
                   const Shader::Profile& profile_, vk::PipelineCache pipeline_cache,
                   bool is_compute_ /*= false*/)
    : instance{instance_}, scheduler{scheduler_}, desc_heap{desc_heap_}, profile{profile_},
      is_compute{is_compute_} {}

Pipeline::~Pipeline() = default;


void Pipeline::BindResources(DescriptorWrites& set_writes,
                             const Shader::PushData& push_data,
                             std::span<const vk::DescriptorBufferInfo> buffer_infos,
                             std::span<const vk::DescriptorImageInfo> image_infos) const {
    const auto bind_point =
        IsCompute() ? vk::PipelineBindPoint::eCompute : vk::PipelineBindPoint::eGraphics;
    const auto stage_flags = IsCompute() ? vk::ShaderStageFlagBits::eCompute : AllGraphicsStageBits;
    const vk::PipelineLayout layout = *pipeline_layout;

    // Bind descriptor set.
    if (set_writes.empty()) {
        scheduler.Record([layout, stage_flags, push_data](vk::CommandBuffer cmdbuf) {
            cmdbuf.pushConstants(layout, stage_flags, 0u, sizeof(push_data), &push_data);
        });
        return;
    }

    if (uses_push_descriptors) {
        if (!scheduler.IsRecordingDeferred()) {
            scheduler.Record([&](vk::CommandBuffer cmdbuf) {
                cmdbuf.pushConstants(layout, stage_flags, 0u, sizeof(push_data), &push_data);
                cmdbuf.pushDescriptorSetKHR(bind_point, layout, 0, set_writes);
            });
            return;
        }
        // Writes and the infos they point to are laid out in the recording chunk, with the
        // pointers already aimed at those copies: the command only carries a span.
        size_t bytes = set_writes.size() * sizeof(vk::WriteDescriptorSet) +
                       buffer_infos.size_bytes() + image_infos.size_bytes() + 64;
        if (buffer_infos.empty() || image_infos.empty()) {
            for (const auto& write : set_writes) {
                if ((write.pBufferInfo && buffer_infos.empty()) ||
                    (write.pImageInfo && image_infos.empty()) || write.pTexelBufferView) {
                    bytes += write.descriptorCount *
                                 (sizeof(vk::DescriptorBufferInfo) + sizeof(vk::DescriptorImageInfo) +
                                  sizeof(vk::BufferView)) +
                             32;
                }
            }
        }
        scheduler.ReserveRecordData(bytes);
        const auto writes = scheduler.RecordData(std::span<const vk::WriteDescriptorSet>{set_writes});
        const auto dst_bufs = scheduler.RecordData(buffer_infos);
        const auto dst_imgs = scheduler.RecordData(image_infos);
        auto* patched = const_cast<vk::WriteDescriptorSet*>(writes.data());
        const auto* src_buf_base = buffer_infos.data();
        const auto* src_img_base = image_infos.data();
        for (size_t i = 0; i < writes.size(); ++i) {
            auto& write = patched[i];
            if (write.pBufferInfo) {
                if (!buffer_infos.empty()) {
                    write.pBufferInfo = dst_bufs.data() + (write.pBufferInfo - src_buf_base);
                } else {
                    write.pBufferInfo =
                        scheduler.RecordData(std::span{write.pBufferInfo, write.descriptorCount}).data();
                }
            }
            if (write.pImageInfo) {
                if (!image_infos.empty()) {
                    write.pImageInfo = dst_imgs.data() + (write.pImageInfo - src_img_base);
                } else {
                    write.pImageInfo =
                        scheduler.RecordData(std::span{write.pImageInfo, write.descriptorCount}).data();
                }
            }
            if (write.pTexelBufferView) {
                write.pTexelBufferView =
                    scheduler.RecordData(std::span{write.pTexelBufferView, write.descriptorCount}).data();
            }
        }
        scheduler.Record([bind_point, layout, stage_flags, push_data, writes](vk::CommandBuffer cmdbuf) {
            cmdbuf.pushConstants(layout, stage_flags, 0u, sizeof(push_data), &push_data);
            cmdbuf.pushDescriptorSetKHR(bind_point, layout, 0, writes);
        });
        return;
    }

    // Descriptor set updates are device calls: they stay on this thread.
    const auto desc_set = desc_heap.Commit(*desc_layout);
    for (auto& set_write : set_writes) {
        set_write.dstSet = desc_set;
    }
    instance.GetDevice().updateDescriptorSets(set_writes, {});
    scheduler.Record([bind_point, layout, stage_flags, push_data, desc_set](vk::CommandBuffer cmdbuf) {
        cmdbuf.pushConstants(layout, stage_flags, 0u, sizeof(push_data), &push_data);
        cmdbuf.bindDescriptorSets(bind_point, layout, 0, desc_set, {});
    });
}

std::string Pipeline::GetDebugString() const {
    std::string stage_desc;
    for (const auto& stage : stages) {
        if (stage) {
            const auto shader_name = PipelineCache::GetShaderName(stage->hw_stage, stage->pgm_hash);
            if (stage_desc.empty()) {
                stage_desc = shader_name;
            } else {
                stage_desc = fmt::format("{},{}", stage_desc, shader_name);
            }
        }
    }
    return stage_desc;
}

} // namespace Vulkan
