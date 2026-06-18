/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * cuFuzzNN - Floating-Point Anomaly Detection Common Definitions
 *
 * Integrates nixnan-style FP anomaly detection with cuFuzz coverage.
 * Detects NaN, Infinity, Subnormal values in FP16/BF16/FP32/FP64 operations.
 */

#ifndef FP_COMMON_H
#define FP_COMMON_H

#include <stdint.h>

/* FP precision types */
typedef enum {
    FP_PREC_NONE = 0,
    FP_PREC_F16  = 1,
    FP_PREC_BF16 = 2,
    FP_PREC_F32  = 3,
    FP_PREC_F64  = 4,
    FP_PREC_TF32 = 5,
} fp_precision_t;

/* FP anomaly types */
typedef enum {
    FP_ANOMALY_NONE     = 0,
    FP_ANOMALY_NAN      = 1,
    FP_ANOMALY_INF_POS  = 2,
    FP_ANOMALY_INF_NEG  = 3,
    FP_ANOMALY_SUBNORM  = 4,
    FP_ANOMALY_DIV_ZERO = 5,
} fp_anomaly_type_t;

/* FP operation types */
typedef enum {
    FP_OP_ARITHMETIC = 0,  /* FADD, FMUL, FFMA, etc. */
    FP_OP_MEMORY     = 1,  /* LD, ST operations */
    FP_OP_CONVERT    = 2,  /* F2F, I2F, F2I conversions */
    FP_OP_COMPARE    = 3,  /* FSETP, etc. */
    FP_OP_SPECIAL    = 4,  /* MUFU (sin, cos, sqrt, rsqrt, etc.) */
} fp_op_type_t;

/* Information collected in the instrumentation function and passed
 * on the channel from the GPU to the CPU */
typedef struct {
    uint64_t grid_launch_id;     /* Kernel launch identifier */
    uint32_t cta_id_x;           /* CTA coordinates */
    uint32_t cta_id_y;
    uint32_t cta_id_z;
    uint32_t warp_id;            /* Warp identifier */
    uint32_t lane_mask;          /* Active lanes with anomalies */
    uint32_t instr_offset;       /* Instruction offset in function */
    uint16_t opcode_id;          /* Opcode identifier */
    uint8_t  precision;          /* fp_precision_t */
    uint8_t  anomaly_type;       /* fp_anomaly_type_t */
    uint8_t  op_type;            /* fp_op_type_t */
    uint8_t  operand_idx;        /* Which operand had the anomaly (0-3) */
    uint8_t  is_output;          /* 1 if anomaly in output, 0 if in input */
    uint8_t  padding;            /* Alignment padding */
    uint64_t func_addr;          /* Function address for source mapping */
} fp_anomaly_t;

/* Statistics structure for tracking anomaly counts */
typedef struct {
    /* Per-precision arithmetic operation anomalies */
    uint64_t nan_count[5];       /* Indexed by fp_precision_t */
    uint64_t nan_repeats[5];
    uint64_t inf_pos_count[5];
    uint64_t inf_pos_repeats[5];
    uint64_t inf_neg_count[5];
    uint64_t inf_neg_repeats[5];
    uint64_t subnorm_count[5];
    uint64_t subnorm_repeats[5];
    uint64_t div_zero_count[5];
    uint64_t div_zero_repeats[5];

    /* Per-precision memory operation anomalies */
    uint64_t mem_nan_count[5];
    uint64_t mem_nan_repeats[5];
    uint64_t mem_inf_count[5];
    uint64_t mem_inf_repeats[5];
    uint64_t mem_subnorm_count[5];
    uint64_t mem_subnorm_repeats[5];
} fp_stats_t;

/* Channel buffer size for FP anomaly reporting */
#define FP_CHANNEL_SIZE (1l << 20)  /* 1 MB */

/* Maximum unique anomalies to track before deduplication */
#define MAX_UNIQUE_ANOMALIES 65536

/* FP32 special value bit patterns */
#define FP32_EXP_MASK      0x7F800000U
#define FP32_MANTISSA_MASK 0x007FFFFFU
#define FP32_SIGN_MASK     0x80000000U
#define FP32_INF_POS       0x7F800000U
#define FP32_INF_NEG       0xFF800000U

/* FP64 special value bit patterns */
#define FP64_EXP_MASK      0x7FF0000000000000ULL
#define FP64_MANTISSA_MASK 0x000FFFFFFFFFFFFFULL
#define FP64_SIGN_MASK     0x8000000000000000ULL

/* FP16 special value bit patterns */
#define FP16_EXP_MASK      0x7C00U
#define FP16_MANTISSA_MASK 0x03FFU
#define FP16_SIGN_MASK     0x8000U

/* BF16 special value bit patterns (same exponent bits as FP32) */
#define BF16_EXP_MASK      0x7F80U
#define BF16_MANTISSA_MASK 0x007FU
#define BF16_SIGN_MASK     0x8000U

/* Device-side helper macros for FP classification
 * Note: For NVBit tools, we define NVBIT_TOOL to enable device functions
 * even when __CUDA_ARCH__ isn't defined at compile time */
#if defined(__CUDA_ARCH__) || defined(NVBIT_TOOL)

/* Check if FP32 value is NaN */
__device__ __forceinline__ bool fp32_is_nan(float val) {
    uint32_t bits = __float_as_uint(val);
    return ((bits & FP32_EXP_MASK) == FP32_EXP_MASK) &&
           ((bits & FP32_MANTISSA_MASK) != 0);
}

/* Check if FP32 value is Infinity (positive or negative) */
__device__ __forceinline__ bool fp32_is_inf(float val) {
    uint32_t bits = __float_as_uint(val);
    return ((bits & FP32_EXP_MASK) == FP32_EXP_MASK) &&
           ((bits & FP32_MANTISSA_MASK) == 0);
}

/* Check if FP32 value is positive infinity */
__device__ __forceinline__ bool fp32_is_inf_pos(float val) {
    return __float_as_uint(val) == FP32_INF_POS;
}

/* Check if FP32 value is negative infinity */
__device__ __forceinline__ bool fp32_is_inf_neg(float val) {
    return __float_as_uint(val) == FP32_INF_NEG;
}

/* Check if FP32 value is subnormal (denormalized) */
__device__ __forceinline__ bool fp32_is_subnormal(float val) {
    uint32_t bits = __float_as_uint(val);
    return ((bits & FP32_EXP_MASK) == 0) &&
           ((bits & FP32_MANTISSA_MASK) != 0);
}

/* Classify FP32 anomaly type */
__device__ __forceinline__ uint8_t fp32_classify_anomaly(float val) {
    uint32_t bits = __float_as_uint(val);
    uint32_t exp = bits & FP32_EXP_MASK;
    uint32_t mant = bits & FP32_MANTISSA_MASK;

    if (exp == FP32_EXP_MASK) {
        if (mant != 0) return FP_ANOMALY_NAN;
        return (bits & FP32_SIGN_MASK) ? FP_ANOMALY_INF_NEG : FP_ANOMALY_INF_POS;
    }
    if (exp == 0 && mant != 0) return FP_ANOMALY_SUBNORM;
    return FP_ANOMALY_NONE;
}

/* Check if FP64 value is NaN */
__device__ __forceinline__ bool fp64_is_nan(double val) {
    uint64_t bits = __double_as_longlong(val);
    return ((bits & FP64_EXP_MASK) == FP64_EXP_MASK) &&
           ((bits & FP64_MANTISSA_MASK) != 0);
}

/* Classify FP64 anomaly type */
__device__ __forceinline__ uint8_t fp64_classify_anomaly(double val) {
    uint64_t bits = __double_as_longlong(val);
    uint64_t exp = bits & FP64_EXP_MASK;
    uint64_t mant = bits & FP64_MANTISSA_MASK;

    if (exp == FP64_EXP_MASK) {
        if (mant != 0) return FP_ANOMALY_NAN;
        return (bits & FP64_SIGN_MASK) ? FP_ANOMALY_INF_NEG : FP_ANOMALY_INF_POS;
    }
    if (exp == 0 && mant != 0) return FP_ANOMALY_SUBNORM;
    return FP_ANOMALY_NONE;
}

/* Check if FP16 value (as uint16_t) has anomaly */
__device__ __forceinline__ uint8_t fp16_classify_anomaly(uint16_t bits) {
    uint16_t exp = bits & FP16_EXP_MASK;
    uint16_t mant = bits & FP16_MANTISSA_MASK;

    if (exp == FP16_EXP_MASK) {
        if (mant != 0) return FP_ANOMALY_NAN;
        return (bits & FP16_SIGN_MASK) ? FP_ANOMALY_INF_NEG : FP_ANOMALY_INF_POS;
    }
    if (exp == 0 && mant != 0) return FP_ANOMALY_SUBNORM;
    return FP_ANOMALY_NONE;
}

/* Check if BF16 value (as uint16_t) has anomaly */
__device__ __forceinline__ uint8_t bf16_classify_anomaly(uint16_t bits) {
    uint16_t exp = bits & BF16_EXP_MASK;
    uint16_t mant = bits & BF16_MANTISSA_MASK;

    if (exp == BF16_EXP_MASK) {
        if (mant != 0) return FP_ANOMALY_NAN;
        return (bits & BF16_SIGN_MASK) ? FP_ANOMALY_INF_NEG : FP_ANOMALY_INF_POS;
    }
    if (exp == 0 && mant != 0) return FP_ANOMALY_SUBNORM;
    return FP_ANOMALY_NONE;
}

#endif /* __CUDA_ARCH__ || NVBIT_TOOL */

/* Precision name strings for reporting */
static const char* fp_precision_names[] = {
    "none", "f16", "bf16", "f32", "f64", "tf32"
};

/* Anomaly type name strings for reporting */
static const char* fp_anomaly_names[] = {
    "none", "NaN", "+Infinity", "-Infinity", "Subnormal", "DivByZero"
};

/* Operation type name strings for reporting */
static const char* fp_op_type_names[] = {
    "arithmetic", "memory", "convert", "compare", "special"
};

#endif /* FP_COMMON_H */
