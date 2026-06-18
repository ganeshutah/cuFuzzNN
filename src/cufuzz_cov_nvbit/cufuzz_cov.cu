/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * cuFuzzNN NVBit Coverage + FP Anomaly Detection Tool
 *
 * Combines:
 *   1. cuFuzz: Device-side edge coverage for AFL++ integration
 *   2. nixnan: FP anomaly detection (NaN/Inf/Subnormal) with channel logging
 *
 * Author: Mohamed Tarek (mtarek@nvidia.com)
 * Extended with nixnan integration for FP anomaly detection
 */

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <sstream>
#include <iomanip>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* nvbit utility functions */
#include "utils/utils.h"

/* channel for FP anomaly reporting */
#include "utils/channel.hpp"

/* FP anomaly data structures */
#include "fp_common.h"

#include <sys/shm.h> //AFL: giving NVBit access to AFL shared memory map

/* kernel id counter, maintained in system memory */
uint32_t kernel_id = 0;

/* execution histogram of basic blocks */
#define MAP_SIZE 65536
#define MAX_BBS MAP_SIZE

// AFL hashing function and AFL unique constant
#include "xxhash.h"
#define HASH_CONST 0xa5b35705

#define MAGIC_VALUE_START 1234 // For persistent mode
#define MAGIC_VALUE_END   5678 // For persistent mode

__managed__ uint32_t *exec_cov_bb; // no uint8_t atomiccas
__managed__ uint8_t *exec_cov_bb_quantized;
uint8_t* merged_cov;
uint8_t* trace_bits;
__managed__ uint64_t *prev_cov_bb; // added for edge coverage
#define MAX_THREADS 32000000 //10752

uint64_t total_bbs = 0;

typedef struct {
    uint32_t offset;
    std::string sass;
} instr_t;
std::vector<std::vector<instr_t>> bbs;

typedef struct {
    uint64_t pc;
    std::vector<int> bb_ids;
} kernel_t;
std::map<std::string, kernel_t> kernels;

/* ============================================================================
 * FP ANOMALY DETECTION STATE (nixnan-style)
 * ============================================================================ */

enum class RecvThreadState {
    WORKING,
    STOP,
    FINISHED,
};

struct CTXstate {
    /* context id */
    int id;

    /* Channel used to communicate FP anomalies from GPU to CPU */
    ChannelDev* channel_dev;
    ChannelHost channel_host;

    /* Thread state control */
    volatile RecvThreadState recv_thread_done = RecvThreadState::STOP;
};

/* map to store context state for FP anomaly channel */
std::unordered_map<CUcontext, CTXstate*> ctx_state_map;

/* opcode to id map for FP operations */
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;

/* FP anomaly statistics */
fp_stats_t fp_stats;

/* Set to track unique anomalies (for deduplication) */
std::set<uint64_t> unique_anomalies;

/* skip flag used to avoid re-entry on the nvbit_callback */
bool skip_callback_flag = false;

/* ============================================================================
 * GLOBAL CONTROL VARIABLES
 * ============================================================================ */

uint32_t start_grid_num = 0;
uint32_t end_grid_num = UINT32_MAX;
int verbose = 0;
int active_from_start = 1;
bool mangled = false;
std::string outfilename = "";
bool serialize_grids = true;
int afl_persistent = 0;

/* FP anomaly detection control */
int fp_detect_enabled = 1;        /* Enable FP anomaly detection */
int fp_detect_nan = 1;            /* Detect NaN values */
int fp_detect_inf = 1;            /* Detect Infinity values */
int fp_detect_subnorm = 0;        /* Detect subnormal values (disabled by default - noisy) */
int fp_verbose = 0;               /* Verbose FP anomaly output */

/* grid launch id, incremented at every launch */
uint64_t grid_launch_id = 0;

/* used to select region of interest when active from start is off */
bool active_region = true;

/* mutexes */
pthread_mutex_t mutex;
pthread_mutex_t cuda_event_mutex;

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

/* ============================================================================
 * FP ANOMALY RECEIVING THREAD
 * ============================================================================ */

void* fp_recv_thread_fun(void* args);

/* flush channel kernel */
__global__ void flush_channel(ChannelDev* ch_dev) { ch_dev->flush(); }

/* ============================================================================
 * INITIALIZATION
 * ============================================================================ */

void nvbit_at_init() {
    /* just make sure all managed variables are allocated on GPU */
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);

    /* Coverage control variables */
    GET_VAR_INT(start_grid_num, "START_GRID_NUM", 0,
                "Beginning of the kernel grid launch interval where to apply "
                "instrumentation");
    GET_VAR_INT(end_grid_num, "END_GRID_NUM", UINT32_MAX,
                "End of the kernel grid launch interval where to apply "
                "instrumentation");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    GET_VAR_INT(
        active_from_start, "ACTIVE_FROM_START", 1,
        "Start instruction counting from start or wait for cuProfilerStart "
        "and cuProfilerStop");
    GET_VAR_INT(mangled, "MANGLED_NAMES", 1,
                "Print kernel names mangled or not");
    GET_VAR_INT(serialize_grids, "SERIALIZE_GRIDS", 1, "Serialize grids");

    GET_VAR_STR(outfilename, "OUT_FILENAME",
                "Output file with execution histogram information");
    GET_VAR_INT(afl_persistent, "COV_PERSISTENT", 0, "Are we using AFL_PERSISTENT mode? 0:no, 1:yes");

    /* FP anomaly detection control variables */
    GET_VAR_INT(fp_detect_enabled, "FP_DETECT", 1, "Enable FP anomaly detection (NaN/Inf)");
    GET_VAR_INT(fp_detect_nan, "FP_DETECT_NAN", 1, "Detect NaN values");
    GET_VAR_INT(fp_detect_inf, "FP_DETECT_INF", 1, "Detect Infinity values");
    GET_VAR_INT(fp_detect_subnorm, "FP_DETECT_SUBNORM", 0, "Detect subnormal values");
    GET_VAR_INT(fp_verbose, "FP_VERBOSE", 0, "Verbose FP anomaly output");

    if (active_from_start == 0) {
        active_region = false;
    }

    /* Initialize FP statistics */
    memset(&fp_stats, 0, sizeof(fp_stats_t));

    std::string pad(100, '-');
    printf("%s\n", pad.c_str());
    printf("cuFuzzNN: Coverage + FP Anomaly Detection Tool\n");
    printf("  FP Detection: %s\n", fp_detect_enabled ? "enabled" : "disabled");
    if (fp_detect_enabled) {
        printf("  - NaN detection: %s\n", fp_detect_nan ? "yes" : "no");
        printf("  - Inf detection: %s\n", fp_detect_inf ? "yes" : "no");
        printf("  - Subnormal detection: %s\n", fp_detect_subnorm ? "yes" : "no");
    }
    printf("%s\n", pad.c_str());

    /* set mutex as recursive */
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&mutex, &attr);
    pthread_mutex_init(&cuda_event_mutex, &attr);

    //AFL: mapping the AFL shared map into our address space
    const char* shm_id_env = getenv("__AFL_SHM_ID");
    if (shm_id_env != NULL && shm_id_env[0] != '\0') {
        if (verbose) printf("CUFUZZ_COV: shm_id_env is: %s\n", shm_id_env);
        int shm_id = atoi(shm_id_env);
        trace_bits = reinterpret_cast<uint8_t*>(shmat(shm_id, NULL, 0));
        if (trace_bits == (void *)-1) {
            fprintf(stderr, "CUFUZZ_COV: shmat failed, using local buffer\n");
            trace_bits = (uint8_t*)malloc(MAP_SIZE);
            memset(trace_bits, 0, MAP_SIZE);
        }
    } else {
        // Standalone mode: allocate local buffer instead of shared memory
        if (verbose) printf("CUFUZZ_COV: No AFL shared memory, using local buffer\n");
        trace_bits = (uint8_t*)malloc(MAP_SIZE);
        memset(trace_bits, 0, MAP_SIZE);
    }
}

/* ============================================================================
 * HELPER FUNCTIONS FOR FP OPCODE DETECTION
 * ============================================================================ */

/* Check if opcode is a floating-point arithmetic operation */
static bool is_fp_arithmetic_opcode(const char* opcode) {
    /* FP32 operations */
    if (strstr(opcode, "FADD") || strstr(opcode, "FMUL") ||
        strstr(opcode, "FFMA") || strstr(opcode, "FMNMX") ||
        strstr(opcode, "FSET") || strstr(opcode, "FSETP") ||
        strstr(opcode, "MUFU")) return true;

    /* FP64 operations */
    if (strstr(opcode, "DADD") || strstr(opcode, "DMUL") ||
        strstr(opcode, "DFMA") || strstr(opcode, "DMNMX") ||
        strstr(opcode, "DSET") || strstr(opcode, "DSETP")) return true;

    /* FP16 operations */
    if (strstr(opcode, "HADD") || strstr(opcode, "HMUL") ||
        strstr(opcode, "HFMA") || strstr(opcode, "HMNMX") ||
        strstr(opcode, "HSET") || strstr(opcode, "HSETP")) return true;

    /* Tensor Core operations (HMMA, IMMA, etc.) */
    if (strstr(opcode, "HMMA") || strstr(opcode, "DMMA")) return true;

    return false;
}

/* Get FP precision from opcode */
static fp_precision_t get_fp_precision(const char* opcode) {
    if (opcode[0] == 'D' || strstr(opcode, "DMMA")) return FP_PREC_F64;
    if (opcode[0] == 'H' || strstr(opcode, "HMMA") || strstr(opcode, ".F16")) return FP_PREC_F16;
    if (strstr(opcode, ".BF16")) return FP_PREC_BF16;
    if (strstr(opcode, ".TF32")) return FP_PREC_TF32;
    return FP_PREC_F32;  /* Default to FP32 */
}

/* Check if this is a memory operation with FP data */
static bool is_fp_memory_opcode(const char* opcode, const Instr* instr) {
    if (strstr(opcode, "LD") || strstr(opcode, "ST") ||
        strstr(opcode, "LDG") || strstr(opcode, "STG") ||
        strstr(opcode, "LDS") || strstr(opcode, "STS")) {
        /* Check if it's a 32-bit or 64-bit load/store (likely FP) */
        /* This is a heuristic - we can't always tell if data is FP */
        return true;
    }
    return false;
}

/* ============================================================================
 * INSTRUMENTATION
 * ============================================================================ */

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    CTXstate* ctx_state = nullptr;
    if (fp_detect_enabled && ctx_state_map.find(ctx) != ctx_state_map.end()) {
        ctx_state = ctx_state_map[ctx];
    }

    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);

    /* add kernel itself to the related function vector */
    related_functions.push_back(func);

    /* iterate on function */
    for (auto f : related_functions) {
        /* get kernel name */
        std::string name = nvbit_get_func_name(ctx, f, mangled);

        /* if function already instrumented, skip */
        if (!already_instrumented.insert(f).second) {
            continue;
        }

        /* Also check the old kernels map for coverage */
        if (kernels.find(name) != kernels.end()) {
            continue;
        }

        /* Get the static control flow graph of instruction */
        const CFG_t &cfg = nvbit_get_CFG(ctx, f);
        if (cfg.is_degenerate) {
            printf(
                "Warning: Function %s is degenerated, we can't compute basic "
                "blocks statically\n",
                name.c_str());
        }

        if (verbose) {
            printf("Function %s\n", name.c_str());
            int cnt = 0;
            for (auto &bb : cfg.bbs) {
                printf("Basic block id %d - num instructions %ld\n", cnt++,
                       bb->instrs.size());
                for (auto &i : bb->instrs) {
                    i->print(" ");
                }
            }
        }

        if (verbose) {
            printf("inspecting %s - number basic blocks %ld\n", name.c_str(),
                   cfg.bbs.size());
        }
        total_bbs += cfg.bbs.size();

        uint64_t func_pc = nvbit_get_func_addr(ctx, f);
        kernels[name] = {func_pc, std::vector<int>()};

        /* Get all instructions for FP instrumentation */
        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);

        /* ================================================================
         * COVERAGE INSTRUMENTATION (per basic block)
         * ================================================================ */
        for (auto &bb : cfg.bbs) {
            int bb_id = bbs.size();
            if (verbose) printf("CUFUZZ_COV: bb_id: %d\n", bb_id);
            Instr *i = bb->instrs[0];

            /* inject coverage recording function */
            nvbit_insert_call(i, "record_coverage_edge_count", IPOINT_BEFORE);

            // use random id per basic block
            XXH64_hash_t fname_hash = XXH64(name.c_str(), name.length(), HASH_CONST);
            unsigned int bb_id_rand = (int)(((fname_hash << 5) + fname_hash) + bb_id);
            if (verbose) printf("CUFUZZ_COV: bb_id_rand: %u\n", bb_id_rand);
            nvbit_add_call_arg_const_val32(i, bb_id_rand);

            /* AFL: add pointer to device-side coverage array */
            nvbit_add_call_arg_const_val64(i, (uint64_t)exec_cov_bb);

            /* add pointer to previous bb array */
            nvbit_add_call_arg_const_val64(i, (uint64_t)prev_cov_bb);

            if (verbose) {
                i->print("Inject count_instr before - ");
            }

            kernels[name].bb_ids.push_back(bb_id);
            bbs.push_back(std::vector<instr_t>());
            for (auto j : bb->instrs) {
                bbs[bb_id].push_back({j->getOffset(), j->getSass()});
            }
        }

        /* ================================================================
         * FP ANOMALY INSTRUMENTATION (per FP instruction)
         * ================================================================ */
        if (fp_detect_enabled && ctx_state != nullptr) {
            uint32_t instr_idx = 0;
            for (auto instr : instrs) {
                const char* opcode = instr->getOpcode();

                /* Check for FP arithmetic operations */
                if (is_fp_arithmetic_opcode(opcode)) {
                    fp_precision_t prec = get_fp_precision(opcode);

                    /* Build opcode ID map */
                    if (opcode_to_id_map.find(opcode) == opcode_to_id_map.end()) {
                        int opcode_id = opcode_to_id_map.size();
                        opcode_to_id_map[opcode] = opcode_id;
                        id_to_opcode_map[opcode_id] = std::string(opcode);
                    }
                    int opcode_id = opcode_to_id_map[opcode];

                    /* Inject FP result check AFTER the instruction */
                    const char* inject_func = nullptr;
                    switch (prec) {
                        case FP_PREC_F32:
                            inject_func = "instrument_fp32_result";
                            break;
                        case FP_PREC_F64:
                            inject_func = "instrument_fp64_result";
                            break;
                        case FP_PREC_F16:
                        case FP_PREC_BF16:
                            inject_func = "instrument_fp16_result";
                            break;
                        default:
                            inject_func = "instrument_fp32_result";
                    }

                    nvbit_insert_call(instr, inject_func, IPOINT_AFTER);
                    nvbit_add_call_arg_guard_pred_val(instr);

                    /* Add destination register value (32-bit for FP32/FP16, use reg pair for FP64) */
                    /* For FP64, we pass the register number and the function will read both regs */
                    nvbit_add_call_arg_reg_val(instr, instr->getOperand(0)->u.reg.num);

                    nvbit_add_call_arg_const_val32(instr, opcode_id);
                    nvbit_add_call_arg_const_val32(instr, instr->getOffset());
                    nvbit_add_call_arg_launch_val64(instr, 0);  /* grid_launch_id */
                    nvbit_add_call_arg_const_val64(instr, func_pc);
                    nvbit_add_call_arg_const_val64(instr, (uint64_t)ctx_state->channel_dev);

                    if (fp_verbose) {
                        printf("cuFuzzNN: Instrumenting FP op: %s (prec=%s) at offset %u\n",
                               opcode, fp_precision_names[prec], instr->getOffset());
                    }
                }

                /* Check for FP memory operations */
                if (is_fp_memory_opcode(opcode, instr) &&
                    instr->getMemorySpace() != InstrType::MemorySpace::NONE &&
                    instr->getMemorySpace() != InstrType::MemorySpace::CONSTANT) {

                    /* Only instrument loads for now */
                    if (strstr(opcode, "LD")) {
                        if (opcode_to_id_map.find(opcode) == opcode_to_id_map.end()) {
                            int opcode_id = opcode_to_id_map.size();
                            opcode_to_id_map[opcode] = opcode_id;
                            id_to_opcode_map[opcode_id] = std::string(opcode);
                        }
                        int opcode_id = opcode_to_id_map[opcode];

                        /* Find memory reference operand */
                        for (int i = 0; i < instr->getNumOperands(); i++) {
                            const InstrType::operand_t* op = instr->getOperand(i);
                            if (op->type == InstrType::OperandType::MREF) {
                                /* Instrument FP32 loads (heuristic: 32-bit loads) */
                                nvbit_insert_call(instr, "instrument_fp32_load", IPOINT_BEFORE);
                                nvbit_add_call_arg_guard_pred_val(instr);
                                nvbit_add_call_arg_mref_addr64(instr, 0);
                                nvbit_add_call_arg_const_val32(instr, opcode_id);
                                nvbit_add_call_arg_const_val32(instr, instr->getOffset());
                                nvbit_add_call_arg_launch_val64(instr, 0);
                                nvbit_add_call_arg_const_val64(instr, func_pc);
                                nvbit_add_call_arg_const_val64(instr, (uint64_t)ctx_state->channel_dev);
                                break;
                            }
                        }
                    }
                }

                instr_idx++;
            }
        }
    }
}

/* ============================================================================
 * COVERAGE HELPER FUNCTIONS
 * ============================================================================ */

uint32_t count_bytes(void *coverage_array, uint32_t size) {
    uint32_t *ptr = (uint32_t *)coverage_array;
    uint32_t  i = ((size + 3) >> 2);
    uint32_t  ret = 0;

    while (i--) {
        uint32_t v = *(ptr++);
        if (!v) { continue; }
        if (v & 0x000000ffU) { ++ret; }
        if (v & 0x0000ff00U) { ++ret; }
        if (v & 0x00ff0000U) { ++ret; }
        if (v & 0xff000000U) { ++ret; }
    }
    return ret;
}

void print_bytes(void *coverage_array, uint32_t size) {
    uint8_t *ptr = (uint8_t *)coverage_array;
    for (uint32_t i = 0; i < size; i++) {
        if (ptr[i] != 0) {
            printf("CUFUZZ_COV: Byte[%d]: %02X\n", i, ptr[i]);
        }
    }
}

uint32_t merge_coverage_byte_quant(void *host_cov, void* device_cov, void* new_cov) {
    uint8_t *ptr_dev = (uint8_t *)device_cov;
    uint8_t *ptr_host = (uint8_t *)host_cov;
    uint8_t *ptr_new = (uint8_t *)new_cov;

    for (int ix = 0; ix < MAP_SIZE; ix++) {
        uint8_t hctr = ptr_host[ix];
        uint32_t dctr = ptr_dev[ix] + hctr;
        uint8_t dctr8 = dctr & 0xff;
        uint8_t ctr8 = dctr == 0 ? 0 :
                        dctr8 == 0 ? 1 : dctr8;
        ptr_new[ix] = ctr8;
    }

    return 1;
}

uint8_t counterToByte(uint32_t counter) {
    uint8_t out_byte = 0;
    if (counter >= 65536) out_byte = 7;
    else if (counter >= 16384) out_byte = 6;
    else if (counter >= 4096) out_byte = 5;
    else if (counter >= 512) out_byte = 4;
    else if (counter >= 3) out_byte = 3;
    else if (counter >= 2) out_byte = 2;
    else if (counter >= 1) out_byte = 1;
    return out_byte;
}

void quantize_device_map(void *out_quantized, void *in_dev){
    uint32_t *ptr_in = (uint32_t *)in_dev;
    uint8_t *ptr_out = (uint8_t *)out_quantized;

    for (int ix = 0; ix < MAP_SIZE; ix++) {
        uint32_t dctr = ptr_in[ix];
        uint8_t ctr8 = 0;
        ctr8 = counterToByte(dctr);
        ptr_out[ix] = ctr8;
    }
}

/* ============================================================================
 * FP ANOMALY RECEIVING THREAD
 * ============================================================================ */

void* fp_recv_thread_fun(void* args) {
    CUcontext ctx = (CUcontext)args;

    pthread_mutex_lock(&mutex);
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];
    ChannelHost* ch_host = &ctx_state->channel_host;
    pthread_mutex_unlock(&mutex);

    char* recv_buffer = (char*)malloc(FP_CHANNEL_SIZE);

    while (ctx_state->recv_thread_done == RecvThreadState::WORKING) {
        uint32_t num_recv_bytes = ch_host->recv(recv_buffer, FP_CHANNEL_SIZE);
        if (num_recv_bytes > 0) {
            uint32_t num_processed_bytes = 0;
            while (num_processed_bytes < num_recv_bytes) {
                fp_anomaly_t* fa = (fp_anomaly_t*)&recv_buffer[num_processed_bytes];

                /* Create unique key for deduplication */
                uint64_t unique_key = (fa->func_addr << 32) |
                                      (fa->instr_offset << 16) |
                                      (fa->precision << 8) |
                                      fa->anomaly_type;

                bool is_new = unique_anomalies.insert(unique_key).second;

                /* Update statistics */
                uint8_t prec = fa->precision;
                if (fa->op_type == FP_OP_MEMORY) {
                    switch (fa->anomaly_type) {
                        case FP_ANOMALY_NAN:
                            if (is_new) fp_stats.mem_nan_count[prec]++;
                            fp_stats.mem_nan_repeats[prec]++;
                            break;
                        case FP_ANOMALY_INF_POS:
                        case FP_ANOMALY_INF_NEG:
                            if (is_new) fp_stats.mem_inf_count[prec]++;
                            fp_stats.mem_inf_repeats[prec]++;
                            break;
                        case FP_ANOMALY_SUBNORM:
                            if (is_new) fp_stats.mem_subnorm_count[prec]++;
                            fp_stats.mem_subnorm_repeats[prec]++;
                            break;
                    }
                } else {
                    switch (fa->anomaly_type) {
                        case FP_ANOMALY_NAN:
                            if (is_new) fp_stats.nan_count[prec]++;
                            fp_stats.nan_repeats[prec]++;
                            break;
                        case FP_ANOMALY_INF_POS:
                            if (is_new) fp_stats.inf_pos_count[prec]++;
                            fp_stats.inf_pos_repeats[prec]++;
                            break;
                        case FP_ANOMALY_INF_NEG:
                            if (is_new) fp_stats.inf_neg_count[prec]++;
                            fp_stats.inf_neg_repeats[prec]++;
                            break;
                        case FP_ANOMALY_SUBNORM:
                            if (is_new) fp_stats.subnorm_count[prec]++;
                            fp_stats.subnorm_repeats[prec]++;
                            break;
                    }
                }

                /* Print if new and verbose */
                if (is_new && fp_verbose) {
                    const char* opcode_str = "unknown";
                    if (id_to_opcode_map.find(fa->opcode_id) != id_to_opcode_map.end()) {
                        opcode_str = id_to_opcode_map[fa->opcode_id].c_str();
                    }
                    printf("#cuFuzzNN: %s [%s] detected in %s of instruction %s at offset 0x%x "
                           "(CTA %d,%d,%d warp %d, lanes 0x%x)\n",
                           fp_anomaly_names[fa->anomaly_type],
                           fp_precision_names[fa->precision],
                           fa->is_output ? "output" : "input",
                           opcode_str,
                           fa->instr_offset,
                           fa->cta_id_x, fa->cta_id_y, fa->cta_id_z,
                           fa->warp_id,
                           fa->lane_mask);
                }

                num_processed_bytes += sizeof(fp_anomaly_t);
            }
        }
    }
    free(recv_buffer);
    ctx_state->recv_thread_done = RecvThreadState::FINISHED;
    return NULL;
}

/* ============================================================================
 * CUDA EVENT CALLBACKS
 * ============================================================================ */

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    pthread_mutex_lock(&cuda_event_mutex);

    if (skip_callback_flag) {
        pthread_mutex_unlock(&cuda_event_mutex);
        return;
    }
    skip_callback_flag = true;

    CTXstate* ctx_state = nullptr;
    if (fp_detect_enabled && ctx_state_map.find(ctx) != ctx_state_map.end()) {
        ctx_state = ctx_state_map[ctx];
    }

    /* Identify all the possible CUDA launch events */
    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel) {
        cuLaunch_params *p = (cuLaunch_params *)params;

        /* Check for cufuzz_notification_kernel (AFL persistent mode signaling) */
        std::string kernel_name = nvbit_get_func_name(ctx, p->f, 1);
        if (verbose) {printf("CUFUZZ_COV: DEBUG - Kernel name: '%s'\n", kernel_name.c_str());}
        if (kernel_name == "_Z26cufuzz_notification_kerneli" && afl_persistent && !is_exit) {
            if (cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchKernel) {
                cuLaunchKernel_params* p_kernel = (cuLaunchKernel_params*)params;

                if (p_kernel->gridDimX == 1 && p_kernel->gridDimY == 1 && p_kernel->gridDimZ == 1 &&
                    p_kernel->blockDimX == 1 && p_kernel->blockDimY == 1 && p_kernel->blockDimZ == 1) {

                    uint32_t magic_value = *((uint32_t*)p_kernel->kernelParams[0]);

                    if (magic_value == MAGIC_VALUE_START) {
                        if (verbose) {printf("CUFUZZ_COV: cufuzz_notification_kernel(1234) detected\n");}
                        memset(trace_bits, 0, MAP_SIZE);
                        trace_bits[0] = 1;
                    } else if (magic_value == MAGIC_VALUE_END) {
                        if (verbose) {printf("CUFUZZ_COV: cufuzz_notification_kernel(5678) detected\n");}

                        quantize_device_map(exec_cov_bb_quantized, exec_cov_bb);
                        merge_coverage_byte_quant((void *)trace_bits, (void*) exec_cov_bb_quantized, (void*)merged_cov);
                        memcpy(trace_bits, merged_cov, MAP_SIZE);
                        memset(exec_cov_bb, 0, sizeof(uint32_t) * MAX_BBS);
                        memset(exec_cov_bb_quantized, 0, MAP_SIZE);
                    }
                }
            }
            skip_callback_flag = false;
            pthread_mutex_unlock(&cuda_event_mutex);
            return;
        }

        if (!is_exit) {
            pthread_mutex_lock(&mutex);
            instrument_function_if_needed(ctx, p->f);

            if (active_from_start) {
                if (kernel_id >= start_grid_num && kernel_id < end_grid_num) {
                    active_region = true;
                } else {
                    active_region = false;
                }
            }

            nvbit_set_at_launch(ctx, p->f, (uint64_t)grid_launch_id);

            if (verbose && (cbid == API_CUDA_cuLaunchKernel_ptsz ||
                cbid == API_CUDA_cuLaunchKernel)) {
                cuLaunchKernel_params* p2 = (cuLaunchKernel_params*)params;
                printf("Entering: Kernel %s - grid launch id %ld - grid %d,%d,%d - block %d,%d,%d\n",
                       nvbit_get_func_name(ctx, p->f, mangled), grid_launch_id,
                       p2->gridDimX, p2->gridDimY, p2->gridDimZ,
                       p2->blockDimX, p2->blockDimY, p2->blockDimZ);
            }

            if (active_region) {
                nvbit_enable_instrumented(ctx, p->f, true);
            } else {
                nvbit_enable_instrumented(ctx, p->f, false);
            }
            pthread_mutex_unlock(&mutex);
        } else {
            if (serialize_grids) {
                CUDA_SAFECALL(cudaDeviceSynchronize());
            }

            /* Flush FP anomaly channel after kernel */
            if (fp_detect_enabled && ctx_state != nullptr) {
                flush_channel<<<1, 1>>>(ctx_state->channel_dev);
                cudaDeviceSynchronize();
            }

            grid_launch_id++;
        }
    } else if (cbid == API_CUDA_cuProfilerStart && is_exit) {
        if (!active_from_start) {
            active_region = true;
        }
    } else if (cbid == API_CUDA_cuProfilerStop && is_exit) {
        if (!active_from_start) {
            active_region = false;
        }
    }

    skip_callback_flag = false;
    pthread_mutex_unlock(&cuda_event_mutex);
}

/* ============================================================================
 * CONTEXT CALLBACKS
 * ============================================================================ */

void nvbit_at_ctx_init(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    if (verbose) {
        printf("cuFuzzNN: STARTING CONTEXT %p\n", ctx);
    }

    /* Create FP anomaly context state */
    if (fp_detect_enabled) {
        assert(ctx_state_map.find(ctx) == ctx_state_map.end());
        CTXstate* ctx_state = new CTXstate;
        ctx_state->id = ctx_state_map.size();
        ctx_state_map[ctx] = ctx_state;
    }
    pthread_mutex_unlock(&mutex);
}

int cnt_ctx = 0;

void nvbit_tool_init(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    if (cnt_ctx == 0) {
        // Device-side AFL coverage map
        cudaMallocManaged(&exec_cov_bb, sizeof(uint32_t) * MAX_BBS);
        memset(exec_cov_bb, 0, sizeof(uint32_t) * MAX_BBS);

        cudaMallocManaged(&exec_cov_bb_quantized, MAP_SIZE);
        memset(exec_cov_bb_quantized, 0, MAP_SIZE);

        merged_cov = (uint8_t*) malloc(MAP_SIZE);
        memset(merged_cov, 0, MAP_SIZE);

        cudaMallocManaged(&prev_cov_bb, sizeof(uint64_t) * MAX_THREADS);
        memset(prev_cov_bb, 0, sizeof(uint64_t) * MAX_THREADS);
    }
    cnt_ctx++;

    /* Initialize FP anomaly channel */
    if (fp_detect_enabled && ctx_state_map.find(ctx) != ctx_state_map.end()) {
        CTXstate* ctx_state = ctx_state_map[ctx];
        ctx_state->recv_thread_done = RecvThreadState::WORKING;
        cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev));
        ctx_state->channel_host.init(ctx_state->id, FP_CHANNEL_SIZE,
                                     ctx_state->channel_dev, fp_recv_thread_fun, ctx);
        nvbit_set_tool_pthread(ctx_state->channel_host.get_thread());
    }

    pthread_mutex_unlock(&mutex);
}

void print_fp_stats() {
    printf("\n#cuFuzzNN: === FP Anomaly Summary ===\n");

    for (int prec = FP_PREC_F16; prec <= FP_PREC_F64; prec++) {
        bool has_anomalies =
            fp_stats.nan_count[prec] || fp_stats.inf_pos_count[prec] ||
            fp_stats.inf_neg_count[prec] || fp_stats.subnorm_count[prec] ||
            fp_stats.mem_nan_count[prec] || fp_stats.mem_inf_count[prec] ||
            fp_stats.mem_subnorm_count[prec];

        if (!has_anomalies) continue;

        printf("#cuFuzzNN: --- %s Operations ---\n", fp_precision_names[prec]);
        if (fp_stats.nan_count[prec])
            printf("#cuFuzzNN: NaN:         %lu (%lu repeats)\n",
                   fp_stats.nan_count[prec], fp_stats.nan_repeats[prec]);
        if (fp_stats.inf_pos_count[prec])
            printf("#cuFuzzNN: +Infinity:   %lu (%lu repeats)\n",
                   fp_stats.inf_pos_count[prec], fp_stats.inf_pos_repeats[prec]);
        if (fp_stats.inf_neg_count[prec])
            printf("#cuFuzzNN: -Infinity:   %lu (%lu repeats)\n",
                   fp_stats.inf_neg_count[prec], fp_stats.inf_neg_repeats[prec]);
        if (fp_stats.subnorm_count[prec])
            printf("#cuFuzzNN: Subnormal:   %lu (%lu repeats)\n",
                   fp_stats.subnorm_count[prec], fp_stats.subnorm_repeats[prec]);

        if (fp_stats.mem_nan_count[prec] || fp_stats.mem_inf_count[prec] || fp_stats.mem_subnorm_count[prec]) {
            printf("#cuFuzzNN: --- %s Memory Operations ---\n", fp_precision_names[prec]);
            if (fp_stats.mem_nan_count[prec])
                printf("#cuFuzzNN: NaN:         %lu (%lu repeats)\n",
                       fp_stats.mem_nan_count[prec], fp_stats.mem_nan_repeats[prec]);
            if (fp_stats.mem_inf_count[prec])
                printf("#cuFuzzNN: Infinity:    %lu (%lu repeats)\n",
                       fp_stats.mem_inf_count[prec], fp_stats.mem_inf_repeats[prec]);
            if (fp_stats.mem_subnorm_count[prec])
                printf("#cuFuzzNN: Subnormal:   %lu (%lu repeats)\n",
                       fp_stats.mem_subnorm_count[prec], fp_stats.mem_subnorm_repeats[prec]);
        }
    }
    printf("#cuFuzzNN: ========================\n\n");
}

void nvbit_at_ctx_term(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    skip_callback_flag = true;

    if (verbose) {
        printf("cuFuzzNN: TERMINATING CONTEXT %p\n", ctx);
    }

    /* Terminate FP anomaly receiving thread */
    if (fp_detect_enabled && ctx_state_map.find(ctx) != ctx_state_map.end()) {
        CTXstate* ctx_state = ctx_state_map[ctx];

        ctx_state->recv_thread_done = RecvThreadState::STOP;
        while (ctx_state->recv_thread_done != RecvThreadState::FINISHED)
            ;

        ctx_state->channel_host.destroy(false);
        cudaFree(ctx_state->channel_dev);
        delete ctx_state;
        ctx_state_map.erase(ctx);
    }

    cnt_ctx--;
    if (cnt_ctx == 0) {
        if(afl_persistent == 0){
            quantize_device_map(exec_cov_bb_quantized, exec_cov_bb);
            uint32_t bytes_set_in_quantized_map = count_bytes((void*)exec_cov_bb_quantized, MAP_SIZE);
            XXH64_hash_t hash_quantized = XXH64(exec_cov_bb_quantized, MAP_SIZE, HASH_CONST);
            merge_coverage_byte_quant((void *)trace_bits, (void*) exec_cov_bb_quantized, (void*)merged_cov);
            if (verbose) {
                fprintf(stdout, "CUFUZZ_COV: device_cov_quan: bytes_set_in_map: %d and hash: 0x%lx\n",
                        bytes_set_in_quantized_map, hash_quantized);
            }
            memcpy(trace_bits, merged_cov, MAP_SIZE);
        }

        /* Print FP anomaly statistics */
        if (fp_detect_enabled) {
            print_fp_stats();
        }

        cudaError_t cudaStat;
        cudaStat = cudaFree(exec_cov_bb);
        if (cudaStat != cudaSuccess) fprintf(stdout, "CUFUZZ_COV: cudaFree exec_cov_bb Failed\n");
        cudaStat = cudaFree(exec_cov_bb_quantized);
        if (cudaStat != cudaSuccess) fprintf(stdout, "CUFUZZ_COV: cudaFree exec_cov_bb_quantized Failed\n");
        free(merged_cov);

        cudaStat = cudaFree(prev_cov_bb);
        if (cudaStat != cudaSuccess) printf("CUFUZZ_COV: cudaFree prev_cov_bb Failed\n");
    }

    skip_callback_flag = false;
    pthread_mutex_unlock(&mutex);
}

void nvbit_at_term() {
    return;
}
