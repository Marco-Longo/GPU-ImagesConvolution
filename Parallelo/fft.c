#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fft.h"

#define PI 3.14159265358979323846

void error(const char *msg)
{
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

cl_event matinit(cl_command_queue que, cl_kernel matinit_k,
    cl_mem d_mat, cl_int nrows, cl_int ncols,
    int n_wait_events, cl_event *wait_events)
{
    cl_int err;
    cl_int arg = 0;
    err = clSetKernelArg(matinit_k, arg++, sizeof(d_mat), &d_mat);
    ocl_check(err, "set matinit arg %d", arg - 1);
    err = clSetKernelArg(matinit_k, arg++, sizeof(nrows), &nrows);
    ocl_check(err, "set matinit arg %d", arg - 1);
    err = clSetKernelArg(matinit_k, arg++, sizeof(ncols), &ncols);
    ocl_check(err, "set matinit arg %d", arg - 1);

    // scelta manuale del work-group size (lws):
    // gws deve essere multiplo di lws in ogni direzione
    const size_t lws[] = { 32, 8 };
    const size_t gws[] = {
        round_mul_up(ncols, lws[0]), //nel kernel get_global_id(0) è l'indice c
        round_mul_up(nrows, lws[1]), //nel kernel get_global_id(1) è l'indice r
    };

    cl_event evt_init;
    err = clEnqueueNDRangeKernel(
        que, matinit_k,
        2, NULL, gws, lws,
        n_wait_events, wait_events, &evt_init);
    ocl_check(err, "enqueue matinit");

    return evt_init;
}
/*
cl_event product(cl_command_queue que, cl_kernel prod_k,
    cl_mem d_v1, cl_mem d_v2, cl_mem d_vprod,
    cl_int nrows, cl_int ncols, // dell'input
    int n_wait_events, cl_event *wait_events)
{
    cl_int err;
    cl_int arg = 0;
    err = clSetKernelArg(prod_k, arg++, sizeof(d_v1), &d_v1);
    ocl_check(err, "set prod arg %d", arg - 1);
    err = clSetKernelArg(prod_k, arg++, sizeof(d_v2), &d_v2);
    ocl_check(err, "set prod arg %d", arg - 1);
    err = clSetKernelArg(prod_k, arg++, sizeof(d_vprod), &d_vprod);
    ocl_check(err, "set prod arg %d", arg - 1);
    err = clSetKernelArg(prod_k, arg++, sizeof(nrows), &nrows);
    ocl_check(err, "set prod arg %d", arg - 1);
    err = clSetKernelArg(prod_k, arg++, sizeof(ncols), &ncols);
    ocl_check(err, "set prod arg %d", arg - 1);

    // scelta manuale del work-group size:
    // gws deve essere multiplo di lws in ogni direzione
    const size_t lws[] = { 32, 8 };
    const size_t gws[] = {
        round_mul_up(ncols, lws[0]),
        round_mul_up(nrows, lws[1]),
    };

    cl_event evt_prod;
    err = clEnqueueNDRangeKernel(
        que, prod_k,
        2, NULL, gws, lws,
        n_wait_events, wait_events, &evt_prod);
    ocl_check(err, "enqueue prod");

    return evt_prod;
}
*/
cl_event fft(cl_command_queue que, cl_kernel fft_k, cl_mem d_src_img,
             cl_mem d_dest_img, int u, int v, int nrows, int ncols, int n_wait_events, cl_event *wait_events)
{
    cl_int err;
    cl_int arg = 0;

    err = clSetKernelArg(fft_k, arg++, sizeof(d_src_img), &d_src_img);
    ocl_check(err, "set fft arg %d", arg - 1);
    err = clSetKernelArg(fft_k, arg++, sizeof(d_dest_img), &d_dest_img);
    ocl_check(err, "set fft arg %d", arg - 1);
    err = clSetKernelArg(fft_k, arg++, sizeof(u), &u);
    ocl_check(err, "set fft arg %d", arg - 1);
    err = clSetKernelArg(fft_k, arg++, sizeof(v), &v);
    ocl_check(err, "set fft arg %d", arg - 1);
    err = clSetKernelArg(fft_k, arg++, sizeof(nrows), &nrows);
    ocl_check(err, "set fft arg %d", arg - 1);
    err = clSetKernelArg(fft_k, arg++, sizeof(ncols), &ncols);
    ocl_check(err, "set fft arg %d", arg - 1);

    const size_t lws[] = {16, 16};
    const size_t gws[] = {
        round_mul_up(ncols, lws[0]),
        round_mul_up(nrows, lws[1]),
    };

    cl_event evt_fft;
    err = clEnqueueNDRangeKernel(que, fft_k, 2, NULL, gws, lws, n_wait_events, wait_events, &evt_fft);
    ocl_check(err, "enqueue fft");

    return evt_fft;
}

cl_event sum(cl_command_queue que, cl_kernel somma_k, cl_mem d_input, cl_mem d_output,
             cl_int numels, cl_int u, cl_int v, int n_wait_events, cl_event *wait_events)
{
    cl_int err;
    cl_int arg = 0;

    err = clSetKernelArg(somma_k, arg++, sizeof(d_input), &d_input);
    ocl_check(err, "set somma arg %d", arg-1);
    err = clSetKernelArg(somma_k, arg++, sizeof(d_output), &d_output);
    ocl_check(err, "set somma arg %d", arg-1);
    err = clSetKernelArg(somma_k, arg++, sizeof(numels), &numels);
    ocl_check(err, "set somma arg %d", arg-1);
    err = clSetKernelArg(somma_k, arg++, sizeof(u), &u);
    ocl_check(err, "set somma arg %d", arg-1);
    err = clSetKernelArg(somma_k, arg++, sizeof(v), &v);
    ocl_check(err, "set somma arg %d", arg-1);

    const size_t lws[] = { 256 };
    const size_t gws[] = { round_mul_up(numels/2, lws[0]) };

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
            if ((atteso[x][y].real-reale[x][y].real)>1.0e-10 || (atteso[x][y].imag-reale[x][y].imag)>1.0e-10)
                fprintf(stderr, "mismatch@ %d %d: %f+%fi != %f+%fi\n", x, y,
                      reale[x][y].real, reale[x][y].imag, atteso[x][y].real, atteso[x][y].imag);
        }
}

int main(int argc, char *argv[])
{
    if (argc != 3)
        error("sintassi: product nrows ncols");

    const int nrows = atoi(argv[1]);
    const int ncols = atoi(argv[2]);
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

    cl_kernel somma_k = clCreateKernel(prog, "somma", &err);
    ocl_check(err, "create kernel somma");

    /* Allocazione buffer */
    cl_mem d_v1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                                 memsize, NULL, &err);
    ocl_check(err, "create buffer v1");
    cl_mem d_vprod = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                    memsize_complex, NULL, &err);
    ocl_check(err, "create buffer vprod");
    cl_mem d_sum = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                  memsize_complex/2, NULL, &err);
    ocl_check(err, "create buffer sum");

    complex **res_fft = init_complex(nrows, ncols);

    cl_event evt_init = matinit(que, matinit_k,
        d_v1, nrows, ncols, 0, NULL);


    cl_event evt_start, evt_end;

    for(int u=0; u<nrows; ++u)
        for(int v=0; v<ncols; ++v)
        {
            cl_event evt_fft;
            if(u==0 && v==0)
            {
                evt_start = fft(que, fft_k, d_v1, d_vprod, u, v, nrows, ncols,
                           1, &evt_init);
                memcpy(&evt_fft, &evt_start, sizeof(cl_event));
            }
            else
                evt_fft = fft(que, fft_k, d_v1, d_vprod, u, v, nrows, ncols,
                           1, &evt_init);

            //RIDUZIONE
            size_t nsums = log(numels)/log(2) + 1;
            cl_event evt_sum[nsums+1];
            cl_mem d_in = d_vprod;
            cl_mem d_out = d_sum;
            cl_int to_reduce = numels;
            evt_sum[0] = evt_fft;
            int i = 0;
            while (to_reduce > 1)
            {
                evt_sum[i+1] = sum(que, somma_k,
                d_in, d_out, to_reduce, u, v, 1, evt_sum + i);
                cl_mem tmp = d_out;
                d_out = d_in;
                d_in = tmp;
                to_reduce /= 2;
                ++i;
            }
            if(u==nrows-1 && v==ncols-1)
                memcpy(&evt_end, &evt_sum[i], sizeof(cl_event));

            err = clFinish(que);
            ocl_check(err, "clFinish");

            //Estrazione risultato
            err = clEnqueueReadBuffer(que, d_in,
                    CL_TRUE, 0, sizeof(complex),
                    &res_fft[u][v], 0, NULL, NULL);

        }

    printf("totale: %gms\t%gGB/s\n", total_runtime_ms(evt_start, evt_end),
          ((2.0*memsize_complex*memsize_complex)/total_runtime_ns(evt_start, evt_end)));

    //verify(res_fft, nrows, ncols);

    clReleaseMemObject(d_v1);
    clReleaseMemObject(d_vprod);
    clReleaseMemObject(d_sum);
    free(res_fft);

    clReleaseKernel(matinit_k);
    clReleaseKernel(fft_k);
    clReleaseKernel(somma_k);
    clReleaseProgram(prog);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);
    return 0;
}
