#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef float     complex_t;
typedef cl_float2 complex;

void error(const char *msg)
{
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

cl_event matinit(cl_command_queue que, cl_kernel matinit_k, cl_mem d_mat, 
                 cl_int nrows, cl_int ncols, int n_wait_events, cl_event *wait_events)
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
    err = clEnqueueNDRangeKernel(que, matinit_k,2, NULL, gws, lws,
                                 n_wait_events, wait_events, &evt_init);
    ocl_check(err, "enqueue matinit");

    return evt_init;
}

cl_event product(cl_command_queue que, cl_kernel prod_k, cl_mem d_v1, 
                 cl_mem d_v2, cl_mem d_vprod, cl_int nrows, cl_int ncols,
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

void verify(const complex *reale, int nrows, int ncols)
{
    for (int r = 0; r < nrows; ++r) {
        for (int c = 0; c < ncols; ++c) 
        {
            //z*w = (a*c - b*d) + (a*d + b*c)*i
            complex atteso = (complex){{((r-c)*(r-c) - (r-c+1)*(r-c+1)), ((r-c)*(r-c+1) + (r-c+1)*(r-c))}};
            if (atteso.s[0] != reale[r*ncols+c].s[0] || atteso.s[1] != reale[r*ncols+c].s[1]) 
            {
                fprintf(stderr, "mismatch @ %d %d: %.9g+%.9gi != %.9g+%.9gi\n",
                        r, c, reale[r*ncols+c].s[0], reale[r*ncols+c].s[1], 
                        atteso.s[0], atteso.s[1]);
                exit(2);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc != 3)
        error("sintassi: product nrows ncols");

    const int nrows = atoi(argv[1]);
    const int ncols = atoi(argv[2]);
    const int numels = nrows*ncols;
    const size_t memsize = numels*sizeof(complex);

    if (nrows <= 0 || ncols <= 0)
        error("nrows, ncols devono essere positivi");

    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p);
    cl_context ctx = create_context(p, d);
    cl_command_queue que = create_queue(ctx, d);
    cl_program prog = create_program("product.ocl", ctx, d);

    cl_int err;

    cl_kernel matinit_k = clCreateKernel(prog, "matinit", &err);
    ocl_check(err, "create kernel matinit");

    cl_kernel product_k = clCreateKernel(prog, "product2", &err);
    ocl_check(err, "create kernel product");

    /* Allocazione buffer */
    cl_mem d_v1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                                 memsize, NULL, &err);
    ocl_check(err, "create buffer v1");
    cl_mem d_v2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                                 memsize, NULL, &err);
    ocl_check(err, "create buffer v2");
    cl_mem d_vprod = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY
                                    | CL_MEM_ALLOC_HOST_PTR, memsize, NULL, &err);
    ocl_check(err, "create buffer vprod");

    cl_event evt_init = matinit(que, matinit_k, d_v1, nrows, ncols, 0, NULL);
    evt_init = matinit(que, matinit_k, d_v2, nrows, ncols, 0, NULL);

    cl_event evt_prod = product(que, product_k, d_v1, d_v2, d_vprod, 
                                nrows, ncols, 1, &evt_init);

    err = clFinish(que);
    ocl_check(err, "clFinish");

    printf("product: %gms\t%gGB/s\n", runtime_ms(evt_prod),
           (3.0*memsize)/runtime_ns(evt_prod));

    //Estrazione risultato
    cl_event evt_map, evt_unmap;
    complex *res = clEnqueueMapBuffer(que, d_vprod, CL_TRUE, CL_MAP_READ,
                                      0, memsize, 0, NULL, &evt_map, &err);
    ocl_check(err, "map buffer vprod");
    printf("map: %gms\t%gGB/s\n", runtime_ms(evt_map),
           (1.0*memsize)/runtime_ns(evt_map));
    verify(res, nrows, ncols);
    err = clEnqueueUnmapMemObject(que, d_vprod, res, 0, NULL, &evt_unmap);
    ocl_check(err, "unmap buffer vprod");
    clFinish(que);
    printf("unmap: %gms\t%gGB/s\n", runtime_ms(evt_unmap),
           (1.0*memsize)/runtime_ns(evt_unmap));

    clReleaseMemObject(d_v1);
    clReleaseMemObject(d_v2);
    clReleaseMemObject(d_vprod);

    clReleaseKernel(matinit_k);
    clReleaseKernel(product_k);
    clReleaseProgram(prog);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);
    return 0;
}

