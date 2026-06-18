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
 * cuFuzz Persistent Mode Sanitizer Wrapper
 * 
 * This wrapper invokes NVIDIA Compute Sanitizer (or ASAN) on fuzzing inputs
 * in AFL++ persistent mode. It is designed to work with AFL++ SAND.
 *
 * Compile-time flags:
 *   -DSAN_MODE_ASAN  : ASAN mode (uses ASAN_APP, exit code 98)
 *   -DSAN_MODE_INIT  : compute-sanitizer initcheck (uses SANITIZER_ARG_INIT, exit code 99)
 *   -DSAN_MODE_RACE  : compute-sanitizer racecheck (uses SANITIZER_ARG_RACE, exit code 99)
 *   (default)        : compute-sanitizer memcheck (uses SANITIZER_ARG, exit code 99)
 *
 * Debug mode: compile with -DDEBUG to enable verbose output
 *
 * Author: Mohamed Tarek (mtarek@nvidia.com)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>

#ifdef DEBUG
#define DEBUG_PRINT(...) fprintf(stderr, __VA_ARGS__)
#else
#define DEBUG_PRINT(...) ((void)0)
#endif

#ifndef __AFL_FUZZ_TESTCASE_LEN
  ssize_t fuzz_len;
  #define __AFL_FUZZ_TESTCASE_LEN fuzz_len
  unsigned char fuzz_buf[1024000];
  #define __AFL_FUZZ_TESTCASE_BUF fuzz_buf
  #define __AFL_FUZZ_INIT() void sync(void);
  #define __AFL_LOOP(x) ((fuzz_len = read(0, fuzz_buf, sizeof(fuzz_buf))) > 0 ? 1 : 0)
  #define __AFL_INIT() sync()
#endif

__AFL_FUZZ_INIT(); // https://github.com/AFLplusplus/AFLplusplus/blob/stable/instrumentation/README.persistent_mode.md

int main(int argc, char *argv[]) {

#ifdef __AFL_HAVE_MANUAL_CONTROL
    __AFL_INIT();
#endif

    // Sanitizer mode is selected at compile-time (see file header for options)
    
#if defined(SAN_MODE_ASAN)
    // ASAN mode: Get the path to the ASAN app
    char *original_app = getenv("ASAN_APP");
    if (original_app == NULL) {
        fprintf(stderr, "Error: ASAN_APP environment variable not set.\n");
        return 1;
    }
    
    // Get the path to the sanitizer from the environment variable
    char *sanitizer_path = getenv("SANITIZER_PATH");
    if (sanitizer_path == NULL) {
        fprintf(stderr, "Error: SANITIZER_PATH environment variable not set.\n");
        return 1;
    }
    
    // Get the path to the sanitizer arguments (ASAN mode uses SANITIZER_ARG_ASAN)
    char *sanitizer_args = getenv("SANITIZER_ARG_ASAN");
    
    // Set the ASAN_OPTIONS environment variable
    if (setenv("ASAN_OPTIONS", "protect_shadow_gap=0:exitcode=98", 1) != 0) {
        perror("Error setting ASAN_OPTIONS");
        return 1;
    }
    DEBUG_PRINT("cuFuzz persistent wrapper: ASAN mode\n");
    
#else
    // Compute-sanitizer modes: Get the original app
    char *original_app = getenv("ORIGINAL_APP");
    if (original_app == NULL) {
        fprintf(stderr, "Error: ORIGINAL_APP environment variable not set.\n");
        return 1;
    }
    
    // Get the path to the sanitizer from the environment variable
    char *sanitizer_path = getenv("SANITIZER_PATH");
    if (sanitizer_path == NULL) {
        fprintf(stderr, "Error: SANITIZER_PATH environment variable not set.\n");
        return 1;
    }
    
    // Get the path to the sanitizer arguments from the environment variable
    char *sanitizer_args;
#if defined(SAN_MODE_INIT)
    DEBUG_PRINT("cuFuzz persistent wrapper: initcheck mode\n");
    sanitizer_args = getenv("SANITIZER_ARG_INIT");
#elif defined(SAN_MODE_RACE)
    DEBUG_PRINT("cuFuzz persistent wrapper: racecheck mode\n");
    sanitizer_args = getenv("SANITIZER_ARG_RACE");
#else
    DEBUG_PRINT("cuFuzz persistent wrapper: memcheck mode\n");
    sanitizer_args = getenv("SANITIZER_ARG");
#endif
    
#endif

    int sanitizer_argc = 0;
    char **sanitizer_argv = NULL;
    if (sanitizer_args != NULL) {
        // Split the sanitizer_args into individual arguments
        char *token = strtok(sanitizer_args, " ");
        while (token != NULL) {
            sanitizer_argv = realloc(sanitizer_argv, (sanitizer_argc + 1) * sizeof(char *));
            sanitizer_argv[sanitizer_argc] = token;
            sanitizer_argc++;
            token = strtok(NULL, " ");
        }
    }

    // Prepare the arguments for execv
    char **new_argv = malloc((sanitizer_argc + argc + 2) * sizeof(char *));
    if (new_argv == NULL) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        return 1;
    }

    new_argv[0] = sanitizer_path; // Set the sanitizer as the first argument
    for (int i = 0; i < sanitizer_argc; i++) {
        new_argv[i + 1] = sanitizer_argv[i]; // Add the sanitizer arguments (if any!)
    }
    new_argv[sanitizer_argc + 1] = original_app; // Set the program path afterwards
    for (int i = 0; i < argc; i++) {
        new_argv[i + sanitizer_argc + 2] = argv[i + 1]; // Copy the remaining default arguments
    }
    new_argv[sanitizer_argc + argc + 1] = NULL; // Null-terminate the argument list

#ifdef DEBUG
    DEBUG_PRINT("cuFuzz persistent wrapper: Launching sanitizer\n");
    DEBUG_PRINT("Arguments:\n");
    for (int i = 0; new_argv[i] != NULL; i++) {
        DEBUG_PRINT("  argv[%d]: %s\n", i, new_argv[i]);
    }
#endif

    // Fork a child process
    pid_t pid = fork();
    if (pid == -1) {
        perror("Error forking process");
        free(new_argv);
        if (sanitizer_argv != NULL) {
            free(sanitizer_argv);
        }
        return 1;
    }

    if (pid == 0) {
        // Child process: execute the sanitizer with the target program
        execv(sanitizer_path, new_argv);
        // If execv returns, there was an error
        perror("Error executing program");
        free(new_argv);
        if (sanitizer_argv != NULL) {
            free(sanitizer_argv);
        }
        exit(1);
    } else {
        // Parent process: wait for the child process to finish
        int status;
        waitpid(pid, &status, 0);
        
#if defined(SAN_MODE_ASAN)
        // Unset the ASAN_OPTIONS environment variable
        unsetenv("ASAN_OPTIONS");
#endif
        
        if (WIFEXITED(status)) {
            int exit_code = WEXITSTATUS(status);
            DEBUG_PRINT("cuFuzz persistent wrapper: Program exited with code: %d\n", exit_code);
#if defined(SAN_MODE_ASAN)
            if (exit_code == 98) {
                // AddressSanitizer detected an error
                abort();
            } else if (exit_code == 99) {
                // Compute Sanitizer detected an error (in ASAN mode, can also run compute-sanitizer)
                abort();
            }
#else
            if (exit_code == 99) {
                // Compute Sanitizer detected an error
                abort();
            }
#endif
        } else {
            fprintf(stderr, "cuFuzz persistent wrapper: Program did not exit normally\n");
        }
    }

    // Cleanup (note: free(NULL) is safe per C standard)
    free(new_argv);
    if (sanitizer_argv != NULL) {
        free(sanitizer_argv);
    }
    return 0;
}
