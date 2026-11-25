// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace AmdGpu {
struct ComputeProgram;
union Regs;
} // namespace AmdGpu

namespace Shader {
struct Info;
}

namespace Vulkan {

class Rasterizer;

/// bbport: shaders whose HLE writes guest memory. They run where the dispatch is processed,
/// not on the draw recording thread: the SRT walkers of the draws after them read that memory.
bool IsGuestMemoryShaderHLE(unsigned long long pgm_hash);

/// True if `ExecuteShaderHLE` can handle this compute shader without binding resources.
bool IsShaderHLE(unsigned long long pgm_hash);

/// True if `ExecuteShaderHLE` for this compute dispatch may write guest memory synchronously
/// (meaning Stage A must drain the draw pipe and run it before walking later draws' SRTs).
bool ShaderHLEMayWriteGuestMemory(const Shader::Info& info,
                                  const AmdGpu::ComputeProgram& cs_program,
                                  Rasterizer& rasterizer);

/// Attempts to execute a shader using HLE if possible.
bool ExecuteShaderHLE(const Shader::Info& info, const AmdGpu::Regs& regs,
                      const AmdGpu::ComputeProgram& cs_program, Rasterizer& rasterizer);

} // namespace Vulkan
