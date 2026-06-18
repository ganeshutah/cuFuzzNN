# cuFuzz Build Progress Report

**Date:** 2026-04-01
**Machine:** beast (RTX 3090, sm_86, Driver 580.126.09, CUDA 13.2)

## Summary

Successfully built cuFuzz GPU fuzzing framework with NVBit v1.7.7.3. All core components are functional. Full AFL++ fuzzing requires `libstdc++-12-dev` to be installed for building instrumented CUDA applications.

## Completed Tasks

### 1. cuFuzz Repository Setup
- Forked cuFuzz from https://github.com/NVlabs/cuFuzz
- Added to parfloat-class as regular directory (not submodule)

### 2. AFL++ Build
- Cloned AFL++ and checked out commit `9cac7ced05eb9f36c1d0b02ad594b3b09cd3938b`
- Applied cuFuzz patch (`AFLplusplus.patch`)
- Built with GCC (not clang) to avoid C++ header issues
- **Status:** ✅ LLVM mode working

### 3. NVBit Setup
- Initially installed NVBit v1.7.5 (crashed on driver 580)
- Upgraded to NVBit v1.7.7.3
- **Status:** ✅ Working on driver 580.126.09

### 4. Coverage Tool (cufuzz_cov.so)
- Built for sm_86 architecture
- Added xxhash source files (xxhash.h, xxhash.c) locally
- Modified Makefile to use local xxhash static library
- **Fixed:** Standalone mode crash - added fallback when `__AFL_SHM_ID` not set
- **Status:** ✅ Working (standalone and AFL++ modes)

### 5. Sanitizer Wrappers
- Built all four wrappers:
  - `wrapper_memcheck.out`
  - `wrapper_initcheck.out`
  - `wrapper_racecheck.out`
  - `wrapper_asan.out`
- **Status:** ✅ All built successfully

### 6. Sample Application Testing
- Built `sampleApp-vanilla.out` with nvcc
- Tested all select values (1-7)
- Confirmed OOB bug detection with compute-sanitizer (select=6)
- **Status:** ✅ Working

## Test Results

### Coverage Tool Test
```bash
$ LD_PRELOAD=cufuzz_cov.so ./sampleApp-vanilla.out in/test3.txt
CUFUZZ_COV: No AFL shared memory, using local buffer
VectorOp select: 3
Test PASSED
Done
```

### NVBit Instruction Count Test
```bash
$ LD_PRELOAD=instr_count.so ./sampleApp-vanilla.out in/test3.txt
kernel 0 - _Z8vectorOpPKfS0_Pfii - #thread-blocks 4, kernel instructions 800
Test PASSED
```

### OOB Bug Detection (compute-sanitizer)
```bash
$ compute-sanitizer --tool=memcheck ./sampleApp-vanilla.out in/test6.txt
Invalid __global__ read of size 4 bytes at vectorOp+0x460
ERROR SUMMARY: 25 errors
```

## Known Issues

### 1. Instrumented Build Requires libstdc++-12-dev
The system has GCC 12 installed but not `libstdc++-12-dev`. Clang-14 selects GCC 12 by default but can't find C++ headers.

**To fix:**
```bash
sudo apt-get install libstdc++-12-dev
```

Then rebuild instrumented sampleApp:
```bash
nvcc sampleApp.cu --compiler-bindir ../../Tools/AFLplusplus/afl-clang-fast++ \
    --gpu-architecture=sm_86 -o sampleApp.out
```

### 2. Full Fuzzing Requires Instrumented Binary
AFL++ fuzzing needs an instrumented binary for the fork server handshake. Currently only vanilla builds work.

## File Locations

| Component | Path |
|-----------|------|
| AFL++ | `cuFuzz/Tools/AFLplusplus/` |
| NVBit | `cuFuzz/Tools/NVBit/` |
| Coverage Tool | `cuFuzz/src/cufuzz_cov_nvbit/cufuzz_cov.so` |
| Sanitizer Wrappers | `cuFuzz/src/cufuzz_sand/wrapper_*.out` |
| Sample App | `cuFuzz/targets/sampleApp/sampleApp-vanilla.out` |
| Test Inputs | `cuFuzz/targets/sampleApp/in/test{3,6,7}.txt` |

## Environment

```
GPU: NVIDIA GeForce RTX 3090 (sm_86)
Driver: 580.126.09
CUDA: 13.2
OS: Ubuntu 22.04 (Linux 6.8.0-106-generic)
GCC: 11.4.0
Clang: 14.0.0
```

## Docker Image

A Docker image has been built with all cuFuzz components pre-installed.

### Image Details
- **Image:** `cufuzz:sm86`
- **Base:** `nvidia/cuda:12.9.0-devel-ubuntu22.04`
- **Size:** 6.32GB (compressed)
- **Saved to:** `cufuzz-sm86.tar.gz` (5.9GB)

### Loading the Docker Image
```bash
# Load from saved file
docker load < cufuzz-sm86.tar.gz

# Run with GPU support
docker run --gpus all -it cufuzz:sm86 /bin/bash
```

### Pushing to Docker Hub
```bash
docker login
docker tag cufuzz:sm86 <your-username>/cufuzz:sm86
docker push <your-username>/cufuzz:sm86
```

### Contents
- AFL++ with cuFuzz patches (LLVM mode)
- NVBit v1.7.7.3 coverage instrumentation
- cufuzz_cov.so coverage tool (sm_86)
- All sanitizer wrappers (memcheck, initcheck, racecheck, asan)
- Sample application and test inputs

## Next Steps

1. Install `libstdc++-12-dev` for instrumented builds
2. Build instrumented sampleApp with afl-clang-fast++
3. Run full cuFuzz fuzzing test
4. Test with more complex CUDA applications
5. Push Docker image to Docker Hub when authenticated

## Git Commits

1. `cc27cee` - Add cuFuzz from NVlabs as regular directory
2. `7709eb1` - Build cuFuzz framework for RTX 3090 (sm_86)
3. `35a751d` - Add test6.txt input for OOB bug testing
4. `9ed01f9` - Upgrade NVBit to v1.7.7.3 for driver 580+ support
5. `190df67` - Fix cufuzz_cov.so to run standalone without AFL++
6. `cdcf338` - Fix build.sh for clean NVBit installation
