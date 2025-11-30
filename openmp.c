#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N_POINTS 300000

typedef struct {
    float x, y, z, w;
} Point4D;

/* Monta matriz 4x4 (linearizada) de escala + translacao */
void build_transform_matrix(float sx, float sy, float sz,
                            float tx, float ty, float tz,
                            float M[16]) {
    for (int i = 0; i < 16; i++) {
        M[i] = 0.0f;
    }

    M[0 * 4 + 0] = sx;
    M[1 * 4 + 1] = sy;
    M[2 * 4 + 2] = sz;
    M[3 * 4 + 3] = 1.0f;

    M[0 * 4 + 3] = tx;
    M[1 * 4 + 3] = ty;
    M[2 * 4 + 3] = tz;
}

/* Versão SEQUENCIAL */
void apply_transform_seq(Point4D *points, int n, const float M[16]) {
    for (int i = 0; i < n; i++) {
        float x = points[i].x;
        float y = points[i].y;
        float z = points[i].z;
        float w = points[i].w;

        points[i].x = M[0] * x + M[1] * y + M[2] * z + M[3] * w;
        points[i].y = M[4] * x + M[5] * y + M[6] * z + M[7] * w;
        points[i].z = M[8] * x + M[9] * y + M[10] * z + M[11] * w;
        points[i].w = M[12] * x + M[13] * y + M[14] * z + M[15] * w;
    }
}

/* Versão PARALELA com OpenMP */
void apply_transform_omp(Point4D *points, int n, const float M[16]) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float x = points[i].x;
        float y = points[i].y;
        float z = points[i].z;
        float w = points[i].w;

        points[i].x = M[0] * x + M[1] * y + M[2] * z + M[3] * w;
        points[i].y = M[4] * x + M[5] * y + M[6] * z + M[7] * w;
        points[i].z = M[8] * x + M[9] * y + M[10] * z + M[11] * w;
        points[i].w = M[12] * x + M[13] * y + M[14] * z + M[15] * w;
    }
}

int main(void) {
    float tx, ty, tz;
    float sx, sy, sz;

    printf("Informe tx, ty, tz (translacao): ");
    if (scanf("%f %f %f", &tx, &ty, &tz) != 3) {
        fprintf(stderr, "Erro ao ler translacao.\n");
        return 1;
    }

    printf("Informe sx, sy, sz (escala): ");
    if (scanf("%f %f %f", &sx, &sy, &sz) != 3) {
        fprintf(stderr, "Erro ao ler escala.\n");
        return 1;
    }

    Point4D *points_seq = (Point4D*) malloc(N_POINTS * sizeof(Point4D));
    Point4D *points_omp = (Point4D*) malloc(N_POINTS * sizeof(Point4D));
    if (!points_seq || !points_omp) {
        fprintf(stderr, "Falha ao alocar memoria.\n");
        return 1;
    }

    /* Gera pontos aleatorios 3D iguais para as duas versões */
    srand((unsigned) time(NULL));
    for (int i = 0; i < N_POINTS; i++) {
        float x = (float) rand() / RAND_MAX * 100.0f;
        float y = (float) rand() / RAND_MAX * 100.0f;
        float z = (float) rand() / RAND_MAX * 100.0f;

        points_seq[i].x = x;
        points_seq[i].y = y;
        points_seq[i].z = z;
        points_seq[i].w = 1.0f;

        points_omp[i] = points_seq[i];
    }

    float M[16];
    build_transform_matrix(sx, sy, sz, tx, ty, tz, M);

    /* Tempo SEQUENCIAL */
    double start_seq = omp_get_wtime();
    apply_transform_seq(points_seq, N_POINTS, M);
    double end_seq = omp_get_wtime();
    double time_seq_ms = (end_seq - start_seq) * 1000.0;

    /* Tempo OpenMP */
    double start_omp = omp_get_wtime();
    apply_transform_omp(points_omp, N_POINTS, M);
    double end_omp = omp_get_wtime();
    double time_omp_ms = (end_omp - start_omp) * 1000.0;

    printf("\nResultados (primeiros 5 pontos - versao OpenMP):\n");
    for (int i = 0; i < 5; i++) {
        printf("P%d' = (%.2f, %.2f, %.2f)\n",
               i, points_omp[i].x, points_omp[i].y, points_omp[i].z);
    }

    printf("\nTempo sequencial: %.3f ms\n", time_seq_ms);
    printf("Tempo paralelo (OpenMP): %.3f ms\n", time_omp_ms);
    if (time_omp_ms > 0.0) {
        printf("Speedup (seq/paralelo) = %.2f x\n", time_seq_ms / time_omp_ms);
    }

    free(points_seq);
    free(points_omp);
    return 0;
}
