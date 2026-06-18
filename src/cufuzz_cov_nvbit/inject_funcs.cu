/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * cuFuzzNN NVBit Injection Functions
 * Device-side instrumentation for:
 *   1. Edge coverage collection (cuFuzz)
 *   2. FP anomaly detection (nixnan-style)
 *
 * Author: Mohamed Tarek (mtarek@nvidia.com)
 * Extended with nixnan integration for FP anomaly detection
 */

#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"
#include "utils/channel.hpp"
#include "fp_common.h"

#define MAP_SIZE 65536
#define MAX_BBS MAP_SIZE

/* FP classification functions are defined in fp_common.h with __CUDA_ARCH__ guard */

/* ============================================================================
 * COVERAGE INSTRUMENTATION (Original cuFuzz)
 * ============================================================================ */

extern "C" __device__ __noinline__ void record_coverage_edge_count(int bb_id,
                                                                   uint64_t p_exec_cov_bb,
                                                                   uint64_t p_prev_cov_bb) {
    /* all the active threads will compute the active mask */
    const int active_mask = __ballot_sync(__activemask(), 1);

    /* each thread will get a lane id (get_lane_id is implemented in
     * utils/utils.h) */
    const int laneid = get_laneid();

    /* get the id of the first active thread */
    const int first_laneid = __ffs(active_mask) - 1;

    /* count all the active thread */
    const int num_threads = __popc(active_mask);

    // get last bb from p_prev_cov_bb[thread_id], XOR it with current bb, and update the p_exec_cov_bb.
    // Also store bb_id to p_prev_cov_bb[thread_id]
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x
          + threadIdx.y * blockDim.x * gridDim.x
          + blockIdx.y * blockDim.x * blockDim.y * gridDim.x
          + threadIdx.z * blockDim.x * blockDim.y * gridDim.x * gridDim.y
          + blockIdx.z * blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y;

    uint64_t* prev_cov_bb = (uint64_t*)p_prev_cov_bb;
    int new_bb_id = (MAP_SIZE/2) + ((prev_cov_bb[tid] ^ bb_id) % (MAP_SIZE/2));
    prev_cov_bb[tid] = bb_id >> 1;

    /* only the first active thread will perform the atomic */
    if (first_laneid == laneid) {
        uint32_t* exec_cov_bb = (uint32_t*)p_exec_cov_bb;
        atomicAdd((unsigned int*)&exec_cov_bb[new_bb_id], 1);
    }
}

/* ============================================================================
 * FP ANOMALY DETECTION (nixnan-style integration)
 * ============================================================================ */

/*
 * Instrument FP32 arithmetic operations (FADD, FMUL, FFMA, etc.)
 * Checks result register for NaN/Inf/Subnormal after operation executes
 */
extern "C" __device__ __noinline__ void instrument_fp32_result(
    int pred,
    uint32_t result_bits,  /* Register value as 32-bit integer */
    uint32_t opcode_id,
    uint32_t instr_offset,
    uint64_t grid_launch_id,
    uint64_t func_addr,
    uint64_t pchannel_dev) {

    if (!pred) return;

    /* Interpret the register bits as float */
    float result = __uint_as_float(result_bits);

    /* Classify the FP32 result */
    uint8_t anomaly = fp32_classify_anomaly(result);
    if (anomaly == FP_ANOMALY_NONE) return;

    /* Collect which lanes have anomalies */
    int active_mask = __ballot_sync(__activemask(), 1);
    int anomaly_mask = __ballot_sync(active_mask, anomaly != FP_ANOMALY_NONE);

    const int laneid = get_laneid();
    const int first_laneid = __ffs(anomaly_mask) - 1;

    /* Only the first lane with an anomaly reports */
    if (first_laneid == laneid) {
        int4 cta = get_ctaid();

        fp_anomaly_t fa;
        fa.grid_launch_id = grid_launch_id;
        fa.cta_id_x = cta.x;
        fa.cta_id_y = cta.y;
        fa.cta_id_z = cta.z;
        fa.warp_id = get_warpid();
        fa.lane_mask = anomaly_mask;
        fa.instr_offset = instr_offset;
        fa.opcode_id = (uint16_t)opcode_id;
        fa.precision = FP_PREC_F32;
        fa.anomaly_type = anomaly;
        fa.op_type = FP_OP_ARITHMETIC;
        fa.operand_idx = 0;  /* Result/output operand */
        fa.is_output = 1;
        fa.func_addr = func_addr;

        ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
        channel_dev->push(&fa, sizeof(fp_anomaly_t));
    }
}

/*
 * Instrument FP64 arithmetic operations (DADD, DMUL, DFMA, etc.)
 * Note: We receive a 32-bit register value, which for FP64 is the lower 32 bits.
 * For simplicity, we check the upper 32 bits pattern to detect NaN/Inf.
 * This is a simplified check - full FP64 would need both register halves.
 */
extern "C" __device__ __noinline__ void instrument_fp64_result(
    int pred,
    uint32_t result_lo32,  /* Lower 32 bits of FP64 result */
    uint32_t opcode_id,
    uint32_t instr_offset,
    uint64_t grid_launch_id,
    uint64_t func_addr,
    uint64_t pchannel_dev) {

    if (!pred) return;

    /* For FP64, the exponent is in bits 52-62 (upper word bits 20-30).
     * Since we only have the lower 32 bits, we can't fully detect FP64 anomalies.
     * Skip FP64 detailed detection for now - would need register pair access. */

    /* Suppress unused parameter warnings */
    (void)result_lo32;
    (void)opcode_id;
    (void)instr_offset;
    (void)grid_launch_id;
    (void)func_addr;
    (void)pchannel_dev;
}

/*
 * Instrument FP16 operations (HADD, HMUL, HFMA, etc.)
 * Takes the raw 16-bit value since CUDA doesn't have native half in old archs
 */
extern "C" __device__ __noinline__ void instrument_fp16_result(
    int pred,
    uint32_t result_lo16,  /* Lower 16 bits contain the half value */
    uint32_t opcode_id,
    uint32_t instr_offset,
    uint64_t grid_launch_id,
    uint64_t func_addr,
    uint64_t pchannel_dev) {

    if (!pred) return;

    uint16_t bits = (uint16_t)(result_lo16 & 0xFFFF);
    uint8_t anomaly = fp16_classify_anomaly(bits);
    if (anomaly == FP_ANOMALY_NONE) return;

    int active_mask = __ballot_sync(__activemask(), 1);
    int anomaly_mask = __ballot_sync(active_mask, anomaly != FP_ANOMALY_NONE);

    const int laneid = get_laneid();
    const int first_laneid = __ffs(anomaly_mask) - 1;

    if (first_laneid == laneid) {
        int4 cta = get_ctaid();

        fp_anomaly_t fa;
        fa.grid_launch_id = grid_launch_id;
        fa.cta_id_x = cta.x;
        fa.cta_id_y = cta.y;
        fa.cta_id_z = cta.z;
        fa.warp_id = get_warpid();
        fa.lane_mask = anomaly_mask;
        fa.instr_offset = instr_offset;
        fa.opcode_id = (uint16_t)opcode_id;
        fa.precision = FP_PREC_F16;
        fa.anomaly_type = anomaly;
        fa.op_type = FP_OP_ARITHMETIC;
        fa.operand_idx = 0;
        fa.is_output = 1;
        fa.func_addr = func_addr;

        ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
        channel_dev->push(&fa, sizeof(fp_anomaly_t));
    }
}

/*
 * Instrument FP32 memory loads - check loaded value for anomalies
 */
extern "C" __device__ __noinline__ void instrument_fp32_load(
    int pred,
    uint64_t addr,
    uint32_t opcode_id,
    uint32_t instr_offset,
    uint64_t grid_launch_id,
    uint64_t func_addr,
    uint64_t pchannel_dev) {

    if (!pred) return;

    /* Read the value being loaded */
    float val = *((float*)addr);
    uint8_t anomaly = fp32_classify_anomaly(val);
    if (anomaly == FP_ANOMALY_NONE) return;

    int active_mask = __ballot_sync(__activemask(), 1);
    int anomaly_mask = __ballot_sync(active_mask, anomaly != FP_ANOMALY_NONE);

    const int laneid = get_laneid();
    const int first_laneid = __ffs(anomaly_mask) - 1;

    if (first_laneid == laneid) {
        int4 cta = get_ctaid();

        fp_anomaly_t fa;
        fa.grid_launch_id = grid_launch_id;
        fa.cta_id_x = cta.x;
        fa.cta_id_y = cta.y;
        fa.cta_id_z = cta.z;
        fa.warp_id = get_warpid();
        fa.lane_mask = anomaly_mask;
        fa.instr_offset = instr_offset;
        fa.opcode_id = (uint16_t)opcode_id;
        fa.precision = FP_PREC_F32;
        fa.anomaly_type = anomaly;
        fa.op_type = FP_OP_MEMORY;
        fa.operand_idx = 0;
        fa.is_output = 0;  /* This is a load (input) */
        fa.func_addr = func_addr;

        ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
        channel_dev->push(&fa, sizeof(fp_anomaly_t));
    }
}

/*
 * Instrument FP64 memory loads
 */
extern "C" __device__ __noinline__ void instrument_fp64_load(
    int pred,
    uint64_t addr,
    uint32_t opcode_id,
    uint32_t instr_offset,
    uint64_t grid_launch_id,
    uint64_t func_addr,
    uint64_t pchannel_dev) {

    if (!pred) return;

    double val = *((double*)addr);
    uint8_t anomaly = fp64_classify_anomaly(val);
    if (anomaly == FP_ANOMALY_NONE) return;

    int active_mask = __ballot_sync(__activemask(), 1);
    int anomaly_mask = __ballot_sync(active_mask, anomaly != FP_ANOMALY_NONE);

    const int laneid = get_laneid();
    const int first_laneid = __ffs(anomaly_mask) - 1;

    if (first_laneid == laneid) {
        int4 cta = get_ctaid();

        fp_anomaly_t fa;
        fa.grid_launch_id = grid_launch_id;
        fa.cta_id_x = cta.x;
        fa.cta_id_y = cta.y;
        fa.cta_id_z = cta.z;
        fa.warp_id = get_warpid();
        fa.lane_mask = anomaly_mask;
        fa.instr_offset = instr_offset;
        fa.opcode_id = (uint16_t)opcode_id;
        fa.precision = FP_PREC_F64;
        fa.anomaly_type = anomaly;
        fa.op_type = FP_OP_MEMORY;
        fa.operand_idx = 0;
        fa.is_output = 0;
        fa.func_addr = func_addr;

        ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
        channel_dev->push(&fa, sizeof(fp_anomaly_t));
    }
}

/*
 * Instrument FP16 memory loads
 */
extern "C" __device__ __noinline__ void instrument_fp16_load(
    int pred,
    uint64_t addr,
    uint32_t opcode_id,
    uint32_t instr_offset,
    uint64_t grid_launch_id,
    uint64_t func_addr,
    uint64_t pchannel_dev) {

    if (!pred) return;

    uint16_t bits = *((uint16_t*)addr);
    uint8_t anomaly = fp16_classify_anomaly(bits);
    if (anomaly == FP_ANOMALY_NONE) return;

    int active_mask = __ballot_sync(__activemask(), 1);
    int anomaly_mask = __ballot_sync(active_mask, anomaly != FP_ANOMALY_NONE);

    const int laneid = get_laneid();
    const int first_laneid = __ffs(anomaly_mask) - 1;

    if (first_laneid == laneid) {
        int4 cta = get_ctaid();

        fp_anomaly_t fa;
        fa.grid_launch_id = grid_launch_id;
        fa.cta_id_x = cta.x;
        fa.cta_id_y = cta.y;
        fa.cta_id_z = cta.z;
        fa.warp_id = get_warpid();
        fa.lane_mask = anomaly_mask;
        fa.instr_offset = instr_offset;
        fa.opcode_id = (uint16_t)opcode_id;
        fa.precision = FP_PREC_F16;
        fa.anomaly_type = anomaly;
        fa.op_type = FP_OP_MEMORY;
        fa.operand_idx = 0;
        fa.is_output = 0;
        fa.func_addr = func_addr;

        ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
        channel_dev->push(&fa, sizeof(fp_anomaly_t));
    }
}

/*
 * Generic instrumentation for checking input operands before FP operations
 * This catches cases where NaN/Inf propagates through computations
 */
extern "C" __device__ __noinline__ void instrument_fp32_input(
    int pred,
    float val,
    uint32_t opcode_id,
    uint32_t instr_offset,
    uint32_t operand_idx,
    uint64_t grid_launch_id,
    uint64_t func_addr,
    uint64_t pchannel_dev) {

    if (!pred) return;

    uint8_t anomaly = fp32_classify_anomaly(val);
    if (anomaly == FP_ANOMALY_NONE) return;

    int active_mask = __ballot_sync(__activemask(), 1);
    int anomaly_mask = __ballot_sync(active_mask, anomaly != FP_ANOMALY_NONE);

    const int laneid = get_laneid();
    const int first_laneid = __ffs(anomaly_mask) - 1;

    if (first_laneid == laneid) {
        int4 cta = get_ctaid();

        fp_anomaly_t fa;
        fa.grid_launch_id = grid_launch_id;
        fa.cta_id_x = cta.x;
        fa.cta_id_y = cta.y;
        fa.cta_id_z = cta.z;
        fa.warp_id = get_warpid();
        fa.lane_mask = anomaly_mask;
        fa.instr_offset = instr_offset;
        fa.opcode_id = (uint16_t)opcode_id;
        fa.precision = FP_PREC_F32;
        fa.anomaly_type = anomaly;
        fa.op_type = FP_OP_ARITHMETIC;
        fa.operand_idx = (uint8_t)operand_idx;
        fa.is_output = 0;  /* Input operand */
        fa.func_addr = func_addr;

        ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
        channel_dev->push(&fa, sizeof(fp_anomaly_t));
    }
}
