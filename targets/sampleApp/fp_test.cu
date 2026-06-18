/* Simple FP anomaly test for cuFuzzNN */
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void generate_nan(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        /* Generate NaN by 0/0 */
        float zero = 0.0f;
        out[i] = zero / zero;
    }
}

__global__ void generate_inf(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        /* Generate Inf by dividing by zero */
        float one = 1.0f;
        float zero = 0.0f;
        out[i] = one / zero;
    }
}

__global__ void normal_computation(float* a, float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i] * 2.0f;
    }
}

int main(int argc, char** argv) {
    int n = 256;
    float *d_a, *d_b, *d_out;
    float *h_out = (float*)malloc(n * sizeof(float));

    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    int test = 0;
    if (argc > 1) test = atoi(argv[1]);

    switch (test) {
        case 1:
            printf("Test 1: Generating NaN values\n");
            generate_nan<<<1, n>>>(d_out, n);
            break;
        case 2:
            printf("Test 2: Generating Infinity values\n");
            generate_inf<<<1, n>>>(d_out, n);
            break;
        default:
            printf("Test 0: Normal computation (no anomalies)\n");
            /* Initialize with normal values */
            float *h_a = (float*)malloc(n * sizeof(float));
            float *h_b = (float*)malloc(n * sizeof(float));
            for (int i = 0; i < n; i++) {
                h_a[i] = (float)i;
                h_b[i] = (float)(i * 2);
            }
            cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);
            normal_computation<<<1, n>>>(d_a, d_b, d_out, n);
            free(h_a);
            free(h_b);
            break;
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    /* Print first few values */
    printf("First 5 output values: ");
    for (int i = 0; i < 5; i++) {
        printf("%f ", h_out[i]);
    }
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    free(h_out);

    return 0;
}
