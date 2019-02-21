#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define _2PI 6.283185307179586476925f
typedef cl_double2 complex;
typedef double     complex_t;

//CODICE SEQUENZIALE
int ** init(int M, int N)
{
  int **m = malloc(M * sizeof(int *));
  for(int i=0; i<M; i++)
  {
    m[i] = malloc(N * sizeof(int));
    for(int j=0; j<N; j++)
      m[i][j] = i+j;
  }

  return m;
}

complex ** init_complex(int M, int N)
{
  complex **m = malloc(M * sizeof(complex *));
  for(int i=0; i<M; i++)
  {
    m[i] = malloc(N * sizeof(complex));
    for(int j=0; j<N; j++)
      m[i][j] = (complex){{0.0, 0.0}};
  }

  return m;
}

complex ** fft_c(int **f, int M, int N)
{
  complex **F = init_complex(M,N);

  for(int u=0; u<M; u++)
    for(int v=0; v<N; v++)
    {
      for(int x=0; x<M; x++)
        for(int y=0; y<N; y++)
        {
          complex_t r = _2PI * (((u*x)/(complex_t)M) + ((v*y)/(complex_t)N));

          F[u][v].s[0] += cos(r)*f[x][y];
          F[u][v].s[1] += -sin(r)*f[x][y];
        }
    }

  return F;
}


//CODICE PARALLELO
void error(const char *msg)
{
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

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
        round_mul_up(ncols, lws[0]), //nel kernel get_global_id(0) è l'indice c
        round_mul_up(nrows, lws[1]), //nel kernel get_global_id(1) è l'indice r
    };

    cl_event evt_init;
    err = clEnqueueNDRangeKernel(que, matinit_k, 2, NULL, gws, lws,
                                 n_wait_events, wait_events, &evt_init);
    ocl_check(err, "enqueue matinit");

    return evt_init;
}
/*
cl_event product(cl_command_queue que, cl_kernel prod_k, cl_mem d_v1,
                 cl_mem d_v2, cl_mem d_v3, cl_int nrows, cl_int ncols, // dell'input
                 int n_wait_events, cl_event *wait_events)
{
    cl_int err;
    cl_int arg = 0;
    err = clSetKernelArg(prod_k, arg++, sizeof(d_v1), &d_v1);
    ocl_check(err, "set prod arg %d", arg - 1);
    err = clSetKernelArg(prod_k, arg++, sizeof(d_v2), &d_v2);
    ocl_check(err, "set prod arg %d", arg - 1);
    err = clSetKernelArg(prod_k, arg++, sizeof(d_v3), &d_v3);
    ocl_check(err, "set prod arg %d", arg - 1);
    err = clSetKernelArg(prod_k, arg++, sizeof(nrows), &nrows);
    ocl_check(err, "set prod arg %d", arg - 1);
    err = clSetKernelArg(prod_k, arg++, sizeof(ncols), &ncols);
    ocl_check(err, "set prod arg %d", arg - 1);

    const size_t lws[] = { 32, 8 };
    const size_t gws[] = {
        round_mul_up(ncols, lws[0]),
        round_mul_up(nrows, lws[1]),
    };

    cl_event evt_prod;
    err = clEnqueueNDRangeKernel(que, prod_k, 2, NULL, gws, lws,
                                 n_wait_events, wait_events, &evt_prod);
    ocl_check(err, "enqueue prod");

    return evt_prod;
}
*/
cl_event fft(cl_command_queue que, cl_kernel fft_k, cl_mem d_src_img,
             cl_mem d_dest_img, int nrows, int ncols,
             int n_wait_events, cl_event *wait_events)
{
    cl_int err;
    cl_int arg = 0;

    err = clSetKernelArg(fft_k, arg++, sizeof(d_src_img), &d_src_img);
    ocl_check(err, "set fft arg %d", arg - 1);
    err = clSetKernelArg(fft_k, arg++, sizeof(d_dest_img), &d_dest_img);
    ocl_check(err, "set fft arg %d", arg - 1);
    err = clSetKernelArg(fft_k, arg++, sizeof(nrows), &nrows);
    ocl_check(err, "set fft arg %d", arg - 1);
    err = clSetKernelArg(fft_k, arg++, sizeof(ncols), &ncols);
    ocl_check(err, "set fft arg %d", arg - 1);

    const size_t lws[] = {16, 16};
    const size_t gws[] = {
        round_mul_up(nrows, lws[0]),
        round_mul_up(ncols, lws[1]),
    };

    cl_event evt_fft;
    err = clEnqueueNDRangeKernel(que, fft_k, 2, NULL, gws, lws, n_wait_events,
                                 wait_events, &evt_fft);
    ocl_check(err, "enqueue fft");

    return evt_fft;
}
/*
cl_event sum(cl_command_queue que, cl_kernel somma_k, cl_mem d_input,
             cl_mem d_output, cl_int numels, size_t _lws, size_t _gws,
             int n_wait_events, cl_event *wait_events)
{
    cl_int err;
    cl_int arg = 0;

    err = clSetKernelArg(somma_k, arg++, sizeof(d_input), &d_input);
    ocl_check(err, "set somma arg %d", arg-1);
    err = clSetKernelArg(somma_k, arg++, sizeof(d_output), &d_output);
    ocl_check(err, "set somma arg %d", arg-1);
    err = clSetKernelArg(somma_k, arg++, _lws*sizeof(complex), NULL);
    ocl_check(err, "set vecsum arg %d", arg-1);
    err = clSetKernelArg(somma_k, arg++, sizeof(numels), &numels);
    ocl_check(err, "set somma arg %d", arg-1);

    const size_t lws[] = { _lws };
    const size_t gws[] = { _gws };

    cl_event evt_sum;

    err = clEnqueueNDRangeKernel(que, somma_k, 1, NULL, gws, lws, n_wait_events,
                                 wait_events, &evt_sum);
    ocl_check(err, "enqueue sum");

    return evt_sum;
}

void verify(complex** reale, int nrows, int ncols)
{
    int** input = init(nrows, ncols);
    complex** atteso = fft_c(input, nrows, ncols);

    for (int x = 0; x < nrows; ++x)
      for (int y = 0; y < ncols; ++y)
      {
        if( ((fabs(atteso[x][y].s[0]-reale[x][y].s[0])) > ((2*FLT_EPSILON)*fabs(atteso[x][y].s[0]))) ||
            ((fabs(atteso[x][y].s[1]-reale[x][y].s[1])) > ((2*FLT_EPSILON)*fabs(atteso[x][y].s[1]))) )
        {
          fprintf(stderr, "mismatch@ %d %d: %.9g+%.9gi != %.9g+%.9gi\n",
                  x, y, reale[x][y].s[0], reale[x][y].s[1], atteso[x][y].s[0],
                  atteso[x][y].s[1]);
        }
      }
}
*/
void verify(complex** reale, int nrows, int ncols)
{
    int** input = init(nrows, ncols);
    complex** atteso = fft_c(input, nrows, ncols);

    for (int x = 0; x < nrows; ++x)
        for (int y = 0; y < ncols; ++y)
        {
            if((atteso[x][y].s[0]-reale[x][y].s[0])>1.0e-8 ||
               (atteso[x][y].s[1]-reale[x][y].s[1])>1.0e-8)
                  fprintf(stderr, "mismatch@ %d %d: %.9g+%.9gi != %.9g+%.9gi\n", x, y,
                          reale[x][y].s[0], reale[x][y].s[1], atteso[x][y].s[0],
                          atteso[x][y].s[1]);
        }
}

int main(int argc, char *argv[])
{
    if (argc != 3)
        error("sintassi: product nrows ncols");

    const int nrows = atoi(argv[1]);
    const int ncols = atoi(argv[2]);
    //const size_t lws0 = atoi(argv[3]);
    //const size_t nwg = atoi(argv[4]);

    const int numels = nrows*ncols;
    const size_t memsize = numels*sizeof(int);
    const size_t memsize_complex = numels*sizeof(complex);

    if (nrows <= 0 || ncols <= 0)
        error("nrows, ncols devono essere positivi");
    if (numels & (numels-1))
        error("numels deve essere una potenza di due");

    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p);
    cl_context ctx = create_context(p, d);
    cl_command_queue que = create_queue(ctx, d);
    cl_program prog = create_program("fft.ocl", ctx, d);

    cl_int err;

    cl_kernel matinit_k = clCreateKernel(prog, "matinit", &err);
    ocl_check(err, "create kernel matinit");

    cl_kernel fft_k = clCreateKernel(prog, "fft", &err);
    ocl_check(err, "create kernel fft");

    //cl_kernel somma_k = clCreateKernel(prog, "somma_lmem", &err);
    //ocl_check(err, "create kernel somma");

    //const size_t gws0 = nwg*lws0;
    //size_t memsize_min = gws0*sizeof(complex);

    /* Allocazione buffer */
    cl_mem d_v1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                                 memsize, NULL, &err);
    ocl_check(err, "create buffer v1");
    cl_mem d_v2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                 memsize_complex, NULL, &err);
    ocl_check(err, "create buffer v2");
    //cl_mem d_vsum = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
    //                               memsize_min, NULL, &err);
    //ocl_check(err, "create buffer sum");

    complex **res_fft = init_complex(nrows, ncols);

    //Inizializzazione
    cl_event evt_init = matinit(que, matinit_k, d_v1, nrows, ncols, 0, NULL);

    //Trasformata
    cl_event evt_fft = fft(que, fft_k, d_v1, d_v2, nrows, ncols,
                           1, &evt_init);

    err = clFinish(que);
    ocl_check(err, "clFinish");

    //Estrazione risultato
    complex *res = malloc(numels*sizeof(complex));
    cl_event evt_read;
    err = clEnqueueReadBuffer(que, d_v2, CL_TRUE, 0,
                              numels*sizeof(complex), res, 0, NULL, &evt_read);

    for(int u=0; u<nrows; ++u)
        for(int v=0; v<ncols; ++v)
        {
            res_fft[u][v].s[0] = res[u*ncols+v].s[0];
            res_fft[u][v].s[1] = res[u*ncols+v].s[1];
        }

    printf("totale: %gms\t%gGB/s\n", total_runtime_ms(evt_fft, evt_read),
           ((memsize_complex*(numels+1.0))/total_runtime_ns(evt_fft, evt_read)));

    verify(res_fft, nrows, ncols);

    clReleaseMemObject(d_v1);
    clReleaseMemObject(d_v2);
    //clReleaseMemObject(d_vsum);
    free(res_fft);

    clReleaseKernel(matinit_k);
    clReleaseKernel(fft_k);
    //clReleaseKernel(somma_k);
    clReleaseProgram(prog);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);
    return 0;
}
