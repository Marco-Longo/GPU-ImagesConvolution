#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

typedef float real;

//Codice sequenziale
real **init(int M, int N)
{
  real **m = malloc(M * sizeof(real *));
  for(int i=0; i<M; i++)
  {
    m[i] = malloc(N * sizeof(real));
    for(int j=0; j<N; j++)
      //m[i][j] = i+j;
      m[i][j] = 10*(N*i+j+1);
  }

  return m;
}

real **init0(int M, int N)
{
  real **m = malloc(M * sizeof(real *));
  for(int i=0; i<M; i++)
  {
    m[i] = malloc(N * sizeof(real));
    for(int j=0; j<N; j++)
      m[i][j] = 0.0;
  }

  return m;
}

real **init_kernel(int M, int N)
{
  real **k = malloc(M * sizeof(real *));
  for(int i=0; i<M; i++)
  {
    k[i] = malloc(N * sizeof(real));
    for(int j=0; j<N; j++)
    {
      if(i==1 && j==1)  k[i][j] = 1.0;
      else  k[i][j] = 0.0;
    }
  }

  return k;
}

int clamp(int x, int minval, int maxval)
{
    return fmin(fmax(x, minval), maxval);
}

real **convoluzione(real **m, int M, int N, real **filter, int f_dim)
{
    real **res = init0(M, N);
    int offset = f_dim/2;

    for(int u=0; u<M; u++)
        for(int v=0; v<N; v++)
        {
            real acc = 0.0;
            for(int x=0; x<f_dim; x++)
                for(int y=0; y<f_dim; y++)
                {
                    int row = clamp((int)(u-offset+x), 0, M-1);
                    int col = clamp((int)(v-offset+y), 0, N-1);
                    acc += filter[x][y] * m[row][col];
                }

            res[u][v] = acc;
        }

    return res;
}

//Codice parallelo
void error(const char *msg)
{
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

//Inizializzione matrice
cl_event matinit(cl_command_queue que, cl_kernel matinit_k, cl_mem d_mat,
                 cl_int nrows, cl_int ncols, int n_wait_events,
                 cl_event *wait_events)
{
    cl_int err;
    cl_int arg = 0;
    err = clSetKernelArg(matinit_k, arg++, sizeof(d_mat), &d_mat);
    ocl_check(err, "set matinit arg %d", arg - 1);
    err = clSetKernelArg(matinit_k, arg++, sizeof(nrows), &nrows);
    ocl_check(err, "set matinit arg %d", arg - 1);
    err = clSetKernelArg(matinit_k, arg++, sizeof(ncols), &ncols);
    ocl_check(err, "set matinit arg %d", arg - 1);

    const size_t lws[] = { 32, 8 };
    const size_t gws[] = {
        round_mul_up(ncols, lws[0]),
        round_mul_up(nrows, lws[1]),
    };

    cl_event evt_init;
    err = clEnqueueNDRangeKernel(que, matinit_k, 2, NULL, gws, lws,
                                 n_wait_events, wait_events, &evt_init);
    ocl_check(err, "enqueue matinit");

    return evt_init;
}

//Inizializzazione kernel
cl_event kerinit(cl_command_queue que, cl_kernel kerinit_k, cl_mem d_mat,
                 cl_int nrows, cl_int ncols, int n_wait_events,
                 cl_event *wait_events)
{
    cl_int err;
    cl_int arg = 0;
    err = clSetKernelArg(kerinit_k, arg++, sizeof(d_mat), &d_mat);
    ocl_check(err, "set kerinit arg %d", arg - 1);
    err = clSetKernelArg(kerinit_k, arg++, sizeof(nrows), &nrows);
    ocl_check(err, "set kerinit arg %d", arg - 1);
    err = clSetKernelArg(kerinit_k, arg++, sizeof(ncols), &ncols);
    ocl_check(err, "set kerinit arg %d", arg - 1);

    const size_t lws[] = { 32, 8 };
    const size_t gws[] = {
                           round_mul_up(ncols, lws[0]),
                           round_mul_up(nrows, lws[1]),
                         };

    cl_event evt_init;
    err = clEnqueueNDRangeKernel(que, kerinit_k, 2, NULL, gws, lws,
                                 n_wait_events, wait_events, &evt_init);
    ocl_check(err, "enqueue kerinit");

    return evt_init;
}

cl_event conv(cl_command_queue que, cl_kernel conv_k, cl_mem d_input,
              cl_int nrows, cl_int ncols, cl_mem d_ker, cl_int f_dim,
              cl_mem d_output, int n_wait_events, cl_event *wait_events)
{
    cl_int err;
    cl_int arg = 0;

    err = clSetKernelArg(conv_k, arg++, sizeof(d_input), &d_input);
    ocl_check(err, "set conv arg %d", arg - 1);
    err = clSetKernelArg(conv_k, arg++, sizeof(nrows), &nrows);
    ocl_check(err, "set conv arg %d", arg - 1);
    err = clSetKernelArg(conv_k, arg++, sizeof(ncols), &ncols);
    ocl_check(err, "set conv arg %d", arg - 1);
    err = clSetKernelArg(conv_k, arg++, sizeof(d_ker), &d_ker);
    ocl_check(err, "set conv arg %d", arg - 1);
    err = clSetKernelArg(conv_k, arg++, sizeof(f_dim), &f_dim);
    ocl_check(err, "set conv arg %d", arg - 1);
    err = clSetKernelArg(conv_k, arg++, sizeof(d_output), &d_output);
    ocl_check(err, "set conv arg %d", arg - 1);

    const size_t lws[] = { 32, 8 };
    const size_t gws[] = {
                           round_mul_up(ncols, lws[0]),
                           round_mul_up(nrows, lws[1]),
                         };

    cl_event evt_conv;
    err = clEnqueueNDRangeKernel(que, conv_k, 2, NULL, gws, lws,
                                 n_wait_events, wait_events, &evt_conv);
    ocl_check(err, "enqueue conv");

    return evt_conv;
}
/*
cl_event conv(cl_command_queue que, cl_kernel conv_k, cl_mem d_input,
              cl_int nrows, cl_int ncols, cl_mem d_ker, cl_int f_dim,
              cl_mem d_output, int n_wait_events, cl_event *wait_events)
{
    cl_int err;
    cl_int arg = 0;

    const size_t lws[] = { 32, 8 };
    const size_t gws[] = {
                           round_mul_up(ncols, lws[0]),
                           round_mul_up(nrows, lws[1]),
                         };

    err = clSetKernelArg(conv_k, arg++, sizeof(d_input), &d_input);
    ocl_check(err, "set conv arg %d", arg - 1);
    err = clSetKernelArg(conv_k, arg++, sizeof(nrows), &nrows);
    ocl_check(err, "set conv arg %d", arg - 1);
    err = clSetKernelArg(conv_k, arg++, sizeof(ncols), &ncols);
    ocl_check(err, "set conv arg %d", arg - 1);
    err = clSetKernelArg(conv_k, arg++, sizeof(d_ker), &d_ker);
    ocl_check(err, "set conv arg %d", arg - 1);
    err = clSetKernelArg(conv_k, arg++, sizeof(f_dim), &f_dim);
    ocl_check(err, "set conv arg %d", arg - 1);
    err = clSetKernelArg(conv_k, arg++, sizeof(d_output), &d_output);
    ocl_check(err, "set conv arg %d", arg - 1);
    err = clSetKernelArg(conv_k, arg++, lws[0]*lws[1]*sizeof(real), NULL);
    ocl_check(err, "set conv arg %d", arg - 1);


    cl_event evt_conv;
    err = clEnqueueNDRangeKernel(que, conv_k, 2, NULL, gws, lws,
                                 n_wait_events, wait_events, &evt_conv);
    ocl_check(err, "enqueue conv");

    return evt_conv;
}
*/
void verify(real *reale, int nrows, int ncols, int f_dim)
{
    real **mat = init(nrows, ncols);
    real **ker = init_kernel(f_dim, f_dim);
    real **atteso = convoluzione(mat, nrows, ncols, ker, f_dim);

    for (int x = 0; x < nrows; ++x)
      for (int y = 0; y < ncols; ++y)
      {
        if( ((fabs(atteso[x][y]-reale[x*ncols+y])) > ((FLT_EPSILON)*fabs(atteso[x][y]))) )
        {
          fprintf(stderr, "mismatch@ %d %d: %.9g != %.9g\n",
                  x, y, reale[x*ncols+y], atteso[x][y]);
        }
      }
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        fprintf(stderr, "Sintassi: %s nrows ncols filter\n", argv[0]);
        exit(1);
    }

    const int nrows = atoi(argv[1]);
    const int ncols = atoi(argv[2]);
    const char* filter = argv[3];

    const int numels = nrows*ncols;
    const size_t memsize = numels*sizeof(real);
    const size_t f_dim = strcmp(filter, "Nbox5") == 0 ? 5 : 3;
    const size_t memsize_ker = f_dim*f_dim*sizeof(real);

    if (numels <= 0)
        error("numels deve essere positivo");

    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p);
    cl_context ctx = create_context(p, d);
    cl_command_queue que = create_queue(ctx, d);
    cl_program prog = create_program("conv.ocl", ctx, d);

    cl_int err;

    cl_kernel matinit_k = clCreateKernel(prog, "matinit", &err);
    ocl_check(err, "create kernel matinit");

    cl_kernel kerinit_k = clCreateKernel(prog, filter, &err);
    ocl_check(err, "create kernel kerinit");

    cl_kernel conv_k = clCreateKernel(prog, "conv", &err);
    ocl_check(err, "create kernel conv");

    /* Allocazione buffer */
    cl_mem d_v1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                                 memsize, NULL, &err);
    ocl_check(err, "create buffer v1");
    cl_mem d_ker = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                                  memsize_ker, NULL, &err);
    ocl_check(err, "create buffer ker");
    cl_mem d_v2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                 memsize, NULL, &err);
    ocl_check(err, "create buffer v2");

    cl_event evt_init[2];
    evt_init[0] = matinit(que, matinit_k, d_v1, nrows, ncols, 0, NULL);
    evt_init[1] = kerinit(que, kerinit_k, d_ker, f_dim, f_dim, 0, NULL);

    cl_event evt_conv = conv(que, conv_k, d_v1, nrows, ncols, d_ker, f_dim,
                             d_v2, 2, evt_init);

    err = clFinish(que);
    ocl_check(err, "clFinish");

    printf("conv: %gms\t%gGB/s\n", runtime_ms(evt_conv),
           (memsize_ker*(numels+1.0)/runtime_ns(evt_conv)));

    cl_event evt_map, evt_unmap;
    real *h_res = clEnqueueMapBuffer(que, d_v2, CL_TRUE, CL_MAP_READ,
                                     0, memsize, 0, NULL, &evt_map, &err);
    ocl_check(err, "map buffer v2");
    printf("map: %gms\t%gGB/s\n", runtime_ms(evt_map),
           (1.0*memsize)/runtime_ns(evt_map));
    verify(h_res, nrows, ncols, f_dim);
    err = clEnqueueUnmapMemObject(que, d_v2, h_res, 0, NULL, &evt_unmap);
    ocl_check(err, "unmap buffer v2");
    clFinish(que);
    printf("unmap: %gms\t%gGB/s\n", runtime_ms(evt_unmap),
           (1.0*memsize)/runtime_ns(evt_unmap));

    clReleaseMemObject(d_v1);
    clReleaseMemObject(d_v2);
    clReleaseMemObject(d_ker);

    clReleaseKernel(matinit_k);
    clReleaseKernel(kerinit_k);
    clReleaseKernel(conv_k);
    clReleaseProgram(prog);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);
    return 0;
}
