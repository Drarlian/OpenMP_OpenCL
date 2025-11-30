#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <CL/cl.h>   // Header do OpenCL

#define N_POINTS 300000

typedef struct {
    float x, y, z, w;
} Point4D;

/* Kernel OpenCL como string */
const char *kernelSource =
"typedef struct {                                                \n"
"    float x;                                                    \n"
"    float y;                                                    \n"
"    float z;                                                    \n"
"    float w;                                                    \n"
"} Point4D;                                                      \n"
"__kernel void apply_transform(__global Point4D *points,         \n"
"                              const int n,                      \n"
"                              __constant float *M)              \n"
"{                                                               \n"
"    int i = get_global_id(0);                                   \n"
"    if (i >= n) return;                                         \n"
"    float x = points[i].x;                                      \n"
"    float y = points[i].y;                                      \n"
"    float z = points[i].z;                                      \n"
"    float w = points[i].w;                                      \n"
"    Point4D p;                                                  \n"
"    p.x = M[0]*x + M[1]*y + M[2]*z + M[3]*w;                    \n"
"    p.y = M[4]*x + M[5]*y + M[6]*z + M[7]*w;                    \n"
"    p.z = M[8]*x + M[9]*y + M[10]*z + M[11]*w;                  \n"
"    p.w = M[12]*x + M[13]*y + M[14]*z + M[15]*w;                \n"
"    points[i] = p;                                              \n"
"}                                                               \n";

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

/* Tempo de parede em ms */
double wall_time_ms(void) {
    return (double) clock() * 1000.0 / CLOCKS_PER_SEC;
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

    Point4D *points = (Point4D*) malloc(N_POINTS * sizeof(Point4D));
    if (!points) {
        fprintf(stderr, "Falha ao alocar memoria na CPU.\n");
        return 1;
    }

    srand((unsigned) time(NULL));
    for (int i = 0; i < N_POINTS; i++) {
        points[i].x = (float) rand() / RAND_MAX * 100.0f;
        points[i].y = (float) rand() / RAND_MAX * 100.0f;
        points[i].z = (float) rand() / RAND_MAX * 100.0f;
        points[i].w = 1.0f;
    }

    float M[16];
    build_transform_matrix(sx, sy, sz, tx, ty, tz, M);

    cl_int err;

    /* 1. Descobre plataforma */
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        fprintf(stderr, "Nao ha plataformas OpenCL disponiveis.\n");
        return 1;
    }
    cl_platform_id platform = NULL;
    clGetPlatformIDs(1, &platform, NULL);

    /* 2. Tenta pegar um dispositivo GPU, se nao tiver, cai para CPU */
    cl_device_id device = NULL;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("Nao encontrei GPU OpenCL, tentando CPU...\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Nao ha dispositivo OpenCL disponivel.\n");
            return 1;
        }
    }

    /* 3. Contexto e fila com profiling para medir tempo do kernel */
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS || !context) {
        fprintf(stderr, "Erro ao criar contexto OpenCL.\n");
        return 1;
    }

    cl_command_queue queue = clCreateCommandQueue(
        context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS || !queue) {
        fprintf(stderr, "Erro ao criar fila de comandos.\n");
        clReleaseContext(context);
        return 1;
    }

    /* 4. Programa e kernel */
    const char *sources[] = { kernelSource };
    size_t lengths[] = { strlen(kernelSource) };

    cl_program program = clCreateProgramWithSource(
        context, 1, sources, lengths, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Erro ao criar programa OpenCL.\n");
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        /* Se der erro, imprime o log de compilacao */
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        char *log = (char*) malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              log_size, log, NULL);
        fprintf(stderr, "Erro ao compilar kernel:\n%s\n", log);
        free(log);

        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "apply_transform", &err);
    if (err != CL_SUCCESS || !kernel) {
        fprintf(stderr, "Erro ao criar kernel.\n");
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    /* 5. Buffers na GPU */
    cl_mem d_points = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        N_POINTS * sizeof(Point4D),
        points,
        &err
    );
    if (err != CL_SUCCESS || !d_points) {
        fprintf(stderr, "Erro ao criar buffer de pontos.\n");
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    cl_mem d_M = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        16 * sizeof(float),
        M,
        &err
    );
    if (err != CL_SUCCESS || !d_M) {
        fprintf(stderr, "Erro ao criar buffer da matriz.\n");
        clReleaseMemObject(d_points);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    /* 6. Argumentos do kernel */
    int n = N_POINTS;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_points);
    err |= clSetKernelArg(kernel, 1, sizeof(int),    &n);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_M);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Erro ao setar argumentos do kernel.\n");
        clReleaseMemObject(d_M);
        clReleaseMemObject(d_points);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    /* 7. Configuracao do grid (globalSize multiplo de localSize) */
    size_t localSize  = 256;
    size_t numGroups  = (N_POINTS + localSize - 1) / localSize;
    size_t globalSize = numGroups * localSize;

    cl_event kernel_event;
    double total_start_ms = wall_time_ms();

    /* 8. Enfileira o kernel */
    err = clEnqueueNDRangeKernel(
        queue, kernel,
        1, NULL,
        &globalSize, &localSize,
        0, NULL, &kernel_event
    );
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Erro ao enfileirar kernel.\n");
        clReleaseMemObject(d_M);
        clReleaseMemObject(d_points);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    clFinish(queue);

    /* 9. Tempo do kernel usando profiling */
    cl_ulong start_ns = 0, end_ns = 0;
    clGetEventProfilingInfo(
        kernel_event, CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong), &start_ns, NULL
    );
    clGetEventProfilingInfo(
        kernel_event, CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong), &end_ns, NULL
    );
    double kernel_ms = (double) (end_ns - start_ns) / 1e6;  // ns -> ms

    /* 10. Copia resultados de volta para CPU */
    err = clEnqueueReadBuffer(
        queue, d_points,
        CL_TRUE,
        0,
        N_POINTS * sizeof(Point4D),
        points,
        0, NULL, NULL
    );
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Erro ao ler buffer de pontos.\n");
        clReleaseMemObject(d_M);
        clReleaseMemObject(d_points);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(points);
        return 1;
    }

    double total_end_ms = wall_time_ms();
    double total_ms = total_end_ms - total_start_ms;

    printf("\nResultados (primeiros 5 pontos - versao OpenCL):\n");
    for (int i = 0; i < 5; i++) {
        printf("P%d' = (%.2f, %.2f, %.2f)\n",
               i, points[i].x, points[i].y, points[i].z);
    }

    printf("\nTempo do kernel na GPU (profiling): %.3f ms\n", kernel_ms);
    printf("Tempo total (transferencias + kernel): %.3f ms\n", total_ms);

    /* 11. Limpeza */
    clReleaseEvent(kernel_event);
    clReleaseMemObject(d_M);
    clReleaseMemObject(d_points);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(points);

    return 0;
}
