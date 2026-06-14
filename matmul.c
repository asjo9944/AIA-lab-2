/*
 * Praktikum 1 – Matrix Multiplication on CPU
 * Requires _POSIX_C_SOURCE >= 199309L for clock_gettime / CLOCK_MONOTONIC.
 * ============================================
 * AI Accelerators (AIA) – Lab Assignment
 *
 * Build:  make
 * Run:    ./matmul
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>
#include <time.h>
#include <pthread.h>

#define num_threads 8
const int num_iterations = 4;
#define JB 128  // Tile size – tune to match your L1/L2 cache

// ============================================================================
// IMPLEMENTATION 1: NAIVE MATRIX MULTIPLICATION  (i-j-k)
// Accesses B with stride N down each column → terrible cache behaviour.
// ============================================================================
void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// IMPLEMENTATION 2: LOOP REORDERING  (i-k-j)
// ----------------------------------------------------------------------------
// Key insight: the innermost j-loop now streams through a full row of B and
// accumulates into a full row of C – both are sequential in memory (stride 1).
// A[i*K+k] is a scalar hoisted outside the j-loop, so it is kept in a register.
// This gives the compiler the best possible opportunity to auto-vectorise the
// inner loop with SIMD (SSE/AVX/NEON) and keeps all three operands cache-hot.
//
// Loop order comparison (row-major storage):
//   i-j-k  (naive) : B accessed column-wise → cache miss on every B load
//   i-k-j  (best)  : A scalar, B & C row-wise → fully vectorisable inner loop
//   j-k-i          : C accessed column-wise → cache miss on every C store
//   k-i-j          : A accessed column-wise → cache miss on every A load
// ============================================================================
void matmul_looporder(const float* A, const float* B, float* C, int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A[i * K + k];          // scalar – stays in register
            const float* B_row = B + k * N;     // pointer to row k of B
            float*       C_row = C + i * N;     // pointer to row i of C
            // Inner loop: C[i][j] += a_ik * B[k][j]
            // Sequential reads from B_row, sequential read-writes to C_row.
            // With -O3 -ffast-math the compiler emits a vectorised loop here.
            
            for (int j = 0; j < N; j++) {
                C_row[j] += a_ik * B_row[j];
            }
        }
    }
}

// ============================================================================
// IMPLEMENTATION 3: LOOP TILING  (i-k-j with JB×JB tiles)
// ----------------------------------------------------------------------------
// Even with the i-k-j ordering, for large N the working set of a full row of B
// (N floats) exceeds the L1 cache.  Tiling cuts the matrix into blocks that
// fit in L1/L2:
//
//   tile footprint ≈ 3 × JB² × 4 bytes
//   JB = 64  → 3 × 64² × 4 = 48 KB   (fits comfortably in a 64 KB L1)
//
// Each (i0,k0,j0) block does a complete mini-matmul on its tile before moving
// on, so every cache line loaded is reused JB times instead of 1 time.
// ============================================================================
void matmul_looptiling(const float* A, const float* B, float* C, int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));
    for (int i0 = 0; i0 < M; i0 += JB) {
        int i_end = i0 + JB < M ? i0 + JB : M;
        for (int k0 = 0; k0 < K; k0 += JB) {
            int k_end = k0 + JB < K ? k0 + JB : K;
            for (int j0 = 0; j0 < N; j0 += JB) {
                int j_end = j0 + JB < N ? j0 + JB : N;
                // ---- micro-kernel: i-k-j inside the tile ----
                for (int i = i0; i < i_end; i++) {
                    for (int k = k0; k < k_end; k++) {
                        float a_ik = A[i * K + k];
                        const float* B_row = B + k * N;
                        float*       C_row = C + i * N;
                        for (int j = j0; j < j_end; j++) {
                            C_row[j] += a_ik * B_row[j];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// IMPLEMENTATION 4: TILED + PTHREADS PARALLELISM
// ----------------------------------------------------------------------------
// Strategy: partition the outer i-tile loop across num_threads workers.
// Each thread owns a non-overlapping strip of rows and performs the full
// tiled k-j computation for those rows → zero false sharing on C (different
// rows), read-only sharing of A and B (safe without locks).
//
// Why Pthreads instead of OpenMP?
//   The template already includes <pthread.h> and some environments may not
//   have the OpenMP runtime.  The structure maps directly onto what an OpenMP
//   "#pragma omp parallel for" would generate.
//   If your toolchain supports OpenMP, the equivalent pragma is:
//     #pragma omp parallel for num_threads(num_threads) schedule(dynamic,1)
//   placed on the outer i0 loop inside matmul_looptiling.
// ============================================================================
typedef struct {
    const float* A;
    const float* B;
    float*       C;
    int M, N, K;
    int i_start;   // first row-tile this thread owns
    int i_end;     // one-past-last row-tile
} thread_args_t;

static void* tiled_worker(void* arg) {
    thread_args_t* t = (thread_args_t*)arg;
    const float* A = t->A;
    const float* B = t->B;
    float*       C = t->C;
    int M = t->M, N = t->N, K = t->K;

    for (int i0 = t->i_start; i0 < t->i_end; i0 += JB) {
        int i_end = i0 + JB < M ? i0 + JB : M;
        for (int k0 = 0; k0 < K; k0 += JB) {
            int k_end = k0 + JB < K ? k0 + JB : K;
            for (int j0 = 0; j0 < N; j0 += JB) {
                int j_end = j0 + JB < N ? j0 + JB : N;
                for (int i = i0; i < i_end; i++) {
                    for (int k = k0; k < k_end; k++) {
                        float a_ik = A[i * K + k];
                        const float* B_row = B + k * N;
                        float*       C_row = C + i * N;
                        for (int j = j0; j < j_end; j++) {
                            C_row[j] += a_ik * B_row[j];
                        }
                    }
                }
            }
        }
    }
    return NULL;
}

void matmul_parallel_ikj(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));

    pthread_t      threads[num_threads];
    thread_args_t  args[num_threads];

    // Divide row range [0, M) into num_threads contiguous strips.
    // We work in units of JB so that each thread starts on a tile boundary.
    int rows_per_thread = ((M + num_threads - 1) / num_threads);
    // Round up to the nearest tile boundary for cleaner partitioning.
    rows_per_thread = ((rows_per_thread + JB - 1) / JB) * JB;

    for (int t = 0; t < num_threads; t++) {
        args[t].A       = A;
        args[t].B       = B;
        args[t].C       = C;
        args[t].M       = M;
        args[t].N       = N;
        args[t].K       = K;
        args[t].i_start = t * rows_per_thread;
        args[t].i_end   = args[t].i_start + rows_per_thread;
        if (args[t].i_end > M) args[t].i_end = M;

        if (args[t].i_start >= M) {
            // No rows left for this thread – spawn a no-op thread.
            args[t].i_end = args[t].i_start;
        }
        pthread_create(&threads[t], NULL, tiled_worker, &args[t]);
    }
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}

// ============================================================================
// Utility: matrix init, timing, GFLOP/s, correctness check
// ============================================================================
void initialize_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++)
        matrix[i] = (float)(rand() % 100);
}

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

double calculate_gflops(int M, int N, int K, double total_time_ms) {
    double flops  = 2.0 * M * N * K;
    double gflops = (flops / (total_time_ms / 1000.0)) / 1e9;
    return gflops;
}

int verify_result(const float* C_ref, const float* C_test,
                  int M, int N, float tolerance) {
    for (int i = 0; i < M * N; i++) {
        if (fabs(C_ref[i] - C_test[i]) > tolerance) {
            printf("  MISMATCH at [%d]: ref=%.4f  test=%.4f\n",
                   i, C_ref[i], C_test[i]);
            return 0;
        }
    }
    return 1;
}

typedef void (*matmul_fn)(const float*, const float*, float*, int, int, int);

double benchmark(matmul_fn fn, const float* A, const float* B,
                 float* C, int M, int N, int K) {
    fn(A, B, C, M, N, K);          // warmup – fills caches, JIT, etc.
    double total = 0.0;
    for (int iter = 0; iter < num_iterations; iter++) {
        double t0 = get_time_ms();
        fn(A, B, C, M, N, K);
        __asm__ __volatile__("" : "+m"(C[0]) : : "memory"); // prevent DCE
        double t1 = get_time_ms();
        total += t1 - t0;
    }
    return total / num_iterations;
}

// ============================================================================
// Main
// ============================================================================
int main(void) {
    srand(42);
    printf("MatMul Benchmark – Square Matrices (avg over %d runs)\n\n",
           num_iterations);

    int sizes[] = {2048};
    int n       = (int)(sizeof(sizes) / sizeof(sizes[0]));

    printf("%-6s  %-14s %-14s %-14s %-14s\n",
           "Size", "Naive", "Reordered", "Tiled", "Parallel");
    printf("%-6s  %-14s %-14s %-14s %-14s\n",
           "------", "--------------", "--------------",
           "--------------", "--------------");

    for (int si = 0; si < n; si++) {
        int M = sizes[si], N = M, K = M;

        float* A     = malloc(M * K * sizeof(float));
        float* B     = malloc(K * N * sizeof(float));
        float* C     = malloc(M * N * sizeof(float));
        float* C_ref = malloc(M * N * sizeof(float));

        initialize_matrix(A, M, K);
        initialize_matrix(B, K, N);

        // Compute reference result once for correctness checks
        memset(C_ref, 0, M * N * sizeof(float));
        matmul_naive(A, B, C_ref, M, N, K);

        // Correctness verification (only on smaller sizes for speed)
        if (M <= 256) {
            memset(C, 0, M * N * sizeof(float));
            matmul_looporder(A, B, C, M, N, K);
            if (!verify_result(C_ref, C, M, N, 1e-2f))
                printf("  [WARN] matmul_looporder mismatch at N=%d\n", M);

            memset(C, 0, M * N * sizeof(float));
            matmul_looptiling(A, B, C, M, N, K);
            if (!verify_result(C_ref, C, M, N, 1e-2f))
                printf("  [WARN] matmul_looptiling mismatch at N=%d\n", M);

            memset(C, 0, M * N * sizeof(float));
            matmul_parallel_ikj(A, B, C, M, N, K);
            if (!verify_result(C_ref, C, M, N, 1e-2f))
                printf("  [WARN] matmul_parallel_ikj mismatch at N=%d\n", M);
        }

        // Benchmarks
        memset(C, 0, M * N * sizeof(float));
        double t_naive    = benchmark(matmul_naive,        A, B, C, M, N, K);
        double g_naive    = calculate_gflops(M, N, K, t_naive);

        memset(C, 0, M * N * sizeof(float));
        double t_reorder  = benchmark(matmul_looporder,    A, B, C, M, N, K);
        double g_reorder  = calculate_gflops(M, N, K, t_reorder);

        memset(C, 0, M * N * sizeof(float));
        double t_tiled    = benchmark(matmul_looptiling,   A, B, C, M, N, K);
        double g_tiled    = calculate_gflops(M, N, K, t_tiled);

        memset(C, 0, M * N * sizeof(float));
        double t_parallel = benchmark(matmul_parallel_ikj, A, B, C, M, N, K);
        double g_parallel = calculate_gflops(M, N, K, t_parallel);

        printf("%-6d  %6.2f GFLOP/s\n",
               M, g_parallel);

        free(A); free(B); free(C); free(C_ref);
    }

    return 0;
}