
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <iterator>
#include <string>
#include <cstdio>

#define SHA1_BLOCK_SIZE 64  
#define SHA1_DIGEST_SIZE 20

#define NONCES_PER_THREAD 512

// SHA1 context structure - holds the state during hashing
typedef struct {
    uint32_t state[5];      // The 5 32-bit words (A, B, C, D, E) that form the hash state
    uint32_t count[2];      // Bit count (lower 32 bits in count[0], upper in count[1])
    uint8_t buffer[64];     // Input buffer to accumulate data before processing
} SHA1_CTX;

__constant__ SHA1_CTX const_base_ctx;
__constant__ uint8_t const_suffix[8];
__constant__ uint32_t const_suffix_len;

#define ROL(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

//SHA1 mixing functions
#define F1(b, c, d) (((b) & (c)) | (~(b) & (d)))
#define F2(b, c, d) ((b) ^ (c) ^ (d))
#define F3(b, c, d) (((b) & (c)) | ((b) & (d)) | ((c) & (d)))

// SHA1 constants
#define K1 0x5A827999
#define K2 0x6ED9EBA1
#define K3 0x8F1BBCDC
#define K4 0xCA62C1D6


void sha1_init_cpu(SHA1_CTX* ctx) {
    ctx->state[0] = 0x67452301;
    ctx->state[1] = 0xEFCDAB89;
    ctx->state[2] = 0x98BADCFE;
    ctx->state[3] = 0x10325476;
    ctx->state[4] = 0xC3D2E1F0;
    ctx->count[0] = 0;
    ctx->count[1] = 0;
}

void sha1_transform_cpu(uint32_t state[5], const uint8_t buffer[64]) {
    uint32_t a, b, c, d, e;
    uint32_t w[80];

    for (int i = 0; i < 16; i++) {
        w[i] = (buffer[i * 4] << 24) | (buffer[i * 4 + 1] << 16) |
            (buffer[i * 4 + 2] << 8) | (buffer[i * 4 + 3]);
    }

    for (int i = 16; i < 80; i++) {
        w[i] = ROL(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    }

    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];

    uint32_t temp;

    for (int i = 0; i < 20; i++) {
        temp = ROL(a, 5) + F1(b, c, d) + e + w[i] + K1;
        e = d;
        d = c;
        c = ROL(b, 30);
        b = a;
        a = temp;
    }

    for (int i = 20; i < 40; i++) {
        temp = ROL(a, 5) + F2(b, c, d) + e + w[i] + K2;
        e = d;
        d = c;
        c = ROL(b, 30);
        b = a;
        a = temp;
    }

    for (int i = 40; i < 60; i++) {
        temp = ROL(a, 5) + F3(b, c, d) + e + w[i] + K3;
        e = d;
        d = c;
        c = ROL(b, 30);
        b = a;
        a = temp;
    }

    for (int i = 60; i < 80; i++) {
        temp = ROL(a, 5) + F2(b, c, d) + e + w[i] + K4;
        e = d;
        d = c;
        c = ROL(b, 30);
        b = a;
        a = temp;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
}

void sha1_update_cpu(SHA1_CTX* ctx, const uint8_t* data, uint32_t len) {
    uint32_t i, j;

    j = (ctx->count[0] >> 3) & 63;

    if ((ctx->count[0] += len << 3) < (len << 3)) {
        ctx->count[1]++;
    }
    ctx->count[1] += (len >> 29);

    if ((j + len) > 63) {
        i = 64 - j;
        for (uint32_t k = 0; k < i; k++) {
            ctx->buffer[j + k] = data[k];
        }
        sha1_transform_cpu(ctx->state, ctx->buffer);

        while (i + 63 < len) {
            sha1_transform_cpu(ctx->state, &data[i]);
            i += 64;
        }
        j = 0;
    }
    else {
        i = 0;
    }

    while (i < len) {
        ctx->buffer[j++] = data[i++];
    }
}

__device__ void sha1_init(SHA1_CTX* ctx) {
    ctx->state[0] = 0x67452301;
    ctx->state[1] = 0xEFCDAB89;
    ctx->state[2] = 0x98BADCFE;
    ctx->state[3] = 0x10325476;
    ctx->state[4] = 0xC3D2E1F0;
    ctx->count[0] = 0;
    ctx->count[1] = 0;
}

__device__ void sha1_transform(uint32_t state[5], const uint8_t buffer[64]) {
    uint32_t a, b, c, d, e;
    uint32_t w[80];

    for (int i = 0; i < 16; i++) {
        w[i] = (buffer[i * 4] << 24) | (buffer[i * 4 + 1] << 16) |
            (buffer[i * 4 + 2] << 8) | (buffer[i * 4 + 3]);
    }

    for (int i = 16; i < 80; i++) {
        w[i] = ROL(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    }

    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];

    uint32_t temp;

    for (int i = 0; i < 20; i++) {
        temp = ROL(a, 5) + F1(b, c, d) + e + w[i] + K1;
        e = d;
        d = c;
        c = ROL(b, 30);
        b = a;
        a = temp;
    }

    for (int i = 20; i < 40; i++) {
        temp = ROL(a, 5) + F2(b, c, d) + e + w[i] + K2;
        e = d;
        d = c;
        c = ROL(b, 30);
        b = a;
        a = temp;
    }

    for (int i = 40; i < 60; i++) {
        temp = ROL(a, 5) + F3(b, c, d) + e + w[i] + K3;
        e = d;
        d = c;
        c = ROL(b, 30);
        b = a;
        a = temp;
    }

    for (int i = 60; i < 80; i++) {
        temp = ROL(a, 5) + F2(b, c, d) + e + w[i] + K4;
        e = d;
        d = c;
        c = ROL(b, 30);
        b = a;
        a = temp;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
}

__device__ void sha1_update(SHA1_CTX* ctx, const uint8_t* data, uint32_t len) {
    uint32_t i, j;

    j = (ctx->count[0] >> 3) & 63;

    if ((ctx->count[0] += len << 3) < (len << 3)) {
        ctx->count[1]++;
    }
    ctx->count[1] += (len >> 29);

    if ((j + len) > 63) {
        i = 64 - j;
        for (uint32_t k = 0; k < i; k++) {
            ctx->buffer[j + k] = data[k];
        }
        sha1_transform(ctx->state, ctx->buffer);

        while (i + 63 < len) {
            sha1_transform(ctx->state, &data[i]);
            i += 64;
        }
        j = 0;
    }
    else {
        i = 0;
    }

    while (i < len) {
        ctx->buffer[j++] = data[i++];
    }
}

__device__ void sha1_final(SHA1_CTX* ctx, uint8_t digest[SHA1_DIGEST_SIZE]) {
    uint8_t finalcount[8];
    uint8_t padding[64];
    uint32_t padlen;

    for (int i = 0; i < 8; i++) {
        finalcount[i] = (uint8_t)((ctx->count[(i >= 4 ? 0 : 1)] >> ((3 - (i & 3)) * 8)) & 255);
    }

    uint32_t count = (ctx->count[0] >> 3) & 0x3f;
    padlen = (count < 56) ? (56 - count) : (120 - count);

    padding[0] = 0x80;
    for (uint32_t i = 1; i < padlen; i++) {
        padding[i] = 0x00;
    }

    sha1_update(ctx, padding, padlen);
    sha1_update(ctx, finalcount, 8);

    for (int i = 0; i < SHA1_DIGEST_SIZE; i++) {
        digest[i] = (uint8_t)((ctx->state[i >> 2] >> ((3 - (i & 3)) * 8)) & 255);
    }
}

__global__ void sha1_kernel(uint8_t* result, const uint8_t* message, uint32_t msg_len) {
    SHA1_CTX ctx;
    sha1_init(&ctx);
    sha1_update(&ctx, message, msg_len);
    sha1_final(&ctx, result);
}

__global__ void find_nonce_kernel(
    uint8_t* result_nonce,
    uint32_t* result_nonce_len,
    int* found_flag,
    uint64_t nonce_start,
    uint64_t total_threads
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t nonce = nonce_start + tid;

    uint8_t hash[SHA1_DIGEST_SIZE];
    SHA1_CTX ctx;

    SHA1_CTX base_ctx = const_base_ctx;

    for (int iter = 0; iter < NONCES_PER_THREAD; iter++) {
        if (*found_flag != 0) {
            return;
        }

        ctx = base_ctx;

        uint32_t nonce_len = 0;
        uint64_t temp_nonce = nonce;

        uint8_t nonce_bytes[8];
        nonce_bytes[0] = (uint8_t)(temp_nonce & 0xFF);
        nonce_bytes[1] = (uint8_t)((temp_nonce >> 8) & 0xFF);
        nonce_bytes[2] = (uint8_t)((temp_nonce >> 16) & 0xFF);
        nonce_bytes[3] = (uint8_t)((temp_nonce >> 24) & 0xFF);
        nonce_bytes[4] = (uint8_t)((temp_nonce >> 32) & 0xFF);
        nonce_bytes[5] = (uint8_t)((temp_nonce >> 40) & 0xFF);
        nonce_bytes[6] = (uint8_t)((temp_nonce >> 48) & 0xFF);
        nonce_bytes[7] = (uint8_t)((temp_nonce >> 56) & 0xFF);

        nonce_len = (temp_nonce == 0) ? 1 :
            (temp_nonce <= 0xFF) ? 1 :
            (temp_nonce <= 0xFFFF) ? 2 :
            (temp_nonce <= 0xFFFFFF) ? 3 :
            (temp_nonce <= 0xFFFFFFFF) ? 4 :
            (temp_nonce <= 0xFFFFFFFFFFULL) ? 5 :
            (temp_nonce <= 0xFFFFFFFFFFFFULL) ? 6 :
            (temp_nonce <= 0xFFFFFFFFFFFFFFULL) ? 7 : 8;

        sha1_update(&ctx, nonce_bytes, nonce_len);
        sha1_final(&ctx, hash);

        uint32_t match = 1;
        for (uint32_t i = 0; i < const_suffix_len; i++) {
            match &= (hash[SHA1_DIGEST_SIZE - const_suffix_len + i] == const_suffix[i]);
        }

        if (match) {
            int old = atomicCAS(found_flag, 0, 1);

            if (old == 0) {
                for (uint32_t i = 0; i < nonce_len; i++) {
                    result_nonce[i] = nonce_bytes[i];
                }
                *result_nonce_len = nonce_len;
            }
            return;
        }

        nonce += total_threads;
    }
}

void test_configuration(int num_blocks, int threads_per_block,
    const uint8_t* data, uint32_t data_len,
    const uint8_t* suffix, uint32_t suffix_len) {
    printf("\nTesting: %d blocks x %d threads = %d total threads\n", num_blocks, threads_per_block, num_blocks * threads_per_block);

    uint8_t* d_result_nonce;
    uint32_t* d_result_nonce_len;
    int* d_found_flag;

    cudaMalloc((void**)&d_result_nonce, 16);
    cudaMalloc((void**)&d_result_nonce_len, sizeof(uint32_t));
    cudaMalloc((void**)&d_found_flag, sizeof(int));


    int zero = 0;
    cudaMemcpy(d_found_flag, &zero, sizeof(int), cudaMemcpyHostToDevice);

    uint64_t total_threads = (uint64_t)num_blocks * threads_per_block;
    uint64_t nonce_start = 0;


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int found = 0;
    int iteration = 0;

    while (!found) {

        find_nonce_kernel <<< num_blocks, threads_per_block >>> (d_result_nonce, d_result_nonce_len, d_found_flag, nonce_start, total_threads);

        cudaDeviceSynchronize();

        cudaMemcpy(&found, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);

        iteration++;

        nonce_start += total_threads * NONCES_PER_THREAD;

        if (iteration % 10000 == 0) {
            printf("Iteration %d: checked %llu nonces so far...\n", iteration, nonce_start);
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    uint8_t result_nonce[16];
    uint32_t result_nonce_len;
    cudaMemcpy(result_nonce, d_result_nonce, 16, cudaMemcpyDeviceToHost);
    cudaMemcpy(&result_nonce_len, d_result_nonce_len, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("✓ Found after %d iterations in %.2f ms\n", iteration, milliseconds);
    printf("  Nonce (%u bytes): ", result_nonce_len);
    for (uint32_t i = 0; i < result_nonce_len; i++) {
        printf("%02X ", result_nonce[i]);
    }
    printf("\n");

    uint8_t verify_message[256];
    memcpy(verify_message, data, data_len);
    memcpy(verify_message + data_len, result_nonce, result_nonce_len);

    uint8_t* d_verify_msg, * d_verify_hash;
    cudaMalloc((void**)&d_verify_msg, data_len + result_nonce_len);
    cudaMalloc((void**)&d_verify_hash, SHA1_DIGEST_SIZE);
    cudaMemcpy(d_verify_msg, verify_message, data_len + result_nonce_len, cudaMemcpyHostToDevice);

    sha1_kernel <<<1, 1 >>> (d_verify_hash, d_verify_msg, data_len + result_nonce_len);
    cudaDeviceSynchronize();

    uint8_t final_hash[SHA1_DIGEST_SIZE];
    cudaMemcpy(final_hash, d_verify_hash, SHA1_DIGEST_SIZE, cudaMemcpyDeviceToHost);

    printf("  Full message (DATA + NONCE): ");
    for (uint32_t i = 0; i < data_len + result_nonce_len; i++) {
        printf("%02x", verify_message[i]);
    }
    printf("\n");

    printf("  Full SHA1 hash: ");
    for (int i = 0; i < SHA1_DIGEST_SIZE; i++) {
        printf("%02x", final_hash[i]);
    }
    printf("\n");

    cudaFree(d_result_nonce);
    cudaFree(d_result_nonce_len);
    cudaFree(d_found_flag);
    cudaFree(d_verify_msg);
    cudaFree(d_verify_hash);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {

    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, find_nonce_kernel);
    printf("Kernel registers per thread: %d\n", attr.numRegs);
    printf("Kernel shared memory per block: %zu bytes\n", attr.sharedSizeBytes);
    printf("Kernel constant memory: %zu bytes\n", attr.constSizeBytes);

    int minGridSize, optimalBlockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optimalBlockSize,
        find_nonce_kernel, 0, 0);
    printf("Recommended block size for max occupancy: %d threads\n", optimalBlockSize);
    printf("Recommended grid size: %d blocks\n", minGridSize);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nGPU: %s\n", prop.name);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);

    const char* test_message = "hello";
    uint32_t msg_len = strlen(test_message);

    uint8_t host_result[SHA1_DIGEST_SIZE];
    uint8_t* device_message, * device_result;

    cudaMalloc((void**)&device_message, msg_len);
    cudaMalloc((void**)&device_result, SHA1_DIGEST_SIZE);
    cudaMemcpy(device_message, test_message, msg_len, cudaMemcpyHostToDevice);

    sha1_kernel << <1, 1 >> > (device_result, device_message, msg_len);
    cudaDeviceSynchronize();
    cudaMemcpy(host_result, device_result, SHA1_DIGEST_SIZE, cudaMemcpyDeviceToHost);

    printf("SHA1(\"%s\") = ", test_message);
    for (int i = 0; i < SHA1_DIGEST_SIZE; i++) {
        printf("%02x", host_result[i]);
    }
    printf("\n");
    printf("Expected    = aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d\n");

    cudaFree(device_message);
    cudaFree(device_result);

    printf("\nNonce Finding Performance Test\n");

    const char* data_str = "thisisatestfornoncefinding";
    uint8_t suffix[] = { 0xAB, 0xAB, 0xAB, 0xAB };

    uint32_t data_len = strlen(data_str);
    uint32_t suffix_len = sizeof(suffix);

    printf("DATA: \"%s\" (%u bytes)\n", data_str, data_len);
    printf("SUFFIX: ");
    for (uint32_t i = 0; i < suffix_len; i++) {
        printf("%02X ", suffix[i]);
    }
    printf("(%u byte%s)\n", suffix_len, suffix_len > 1 ? "s" : "");
    printf("Nonces per thread: %d\n", NONCES_PER_THREAD);

    // Pre-compute SHA1 context on CPU
    SHA1_CTX host_base_ctx;
    sha1_init_cpu(&host_base_ctx);
    sha1_update_cpu(&host_base_ctx, (const uint8_t*)data_str, data_len);

    cudaMemcpyToSymbol(const_base_ctx, &host_base_ctx, sizeof(SHA1_CTX));
    cudaMemcpyToSymbol(const_suffix, suffix, suffix_len);
    cudaMemcpyToSymbol(const_suffix_len, &suffix_len, sizeof(uint32_t));

    //Found after 85 iterations in 22878.57 ms
    //test_configuration(128, 128, (const uint8_t*)data_str, data_len, suffix, suffix_len);

    test_configuration(minGridSize, optimalBlockSize, (const uint8_t*)data_str, data_len, suffix, suffix_len);

    // Found after 22 iterations in 15366.35 ms
    //test_configuration(256, 256, (const uint8_t*)data_str, data_len, suffix, suffix_len);

    test_configuration(512, 512, (const uint8_t*)data_str, data_len, suffix, suffix_len);

    test_configuration(350, 512, (const uint8_t*)data_str, data_len, suffix, suffix_len);

    return 0;
}