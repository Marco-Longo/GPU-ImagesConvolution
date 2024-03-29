#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"
#include "pamalign.h"
#include <stdio.h>
#include <stdlib.h>

typedef float     real;
typedef cl_float2 complex;

void error(const char *msg)
{
    fprintf(stderr, "%s\n", msg);
    exit(1);
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
              cl_mem d_ker, cl_int f_dim, cl_mem d_output, int n_wait_events,
              cl_event *wait_events)
{
    cl_int err;
    cl_int arg = 0;
    size_t ncols, nrows;

    err = clGetImageInfo(d_input, CL_IMAGE_WIDTH, sizeof(ncols),
                         &ncols, NULL);
    ocl_check(err, "get image width");

    err = clGetImageInfo(d_input, CL_IMAGE_HEIGHT, sizeof(nrows),
                         &nrows, NULL);
    ocl_check(err, "get image height");


    err = clSetKernelArg(conv_k, arg++, sizeof(d_input), &d_input);
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
              cl_mem d_ker, cl_int f_dim, cl_mem d_output, int n_wait_events,
              cl_event *wait_events)
{
    cl_int err;
    cl_int arg = 0;
    size_t ncols, nrows;

    err = clGetImageInfo(d_input, CL_IMAGE_WIDTH, sizeof(ncols),
                         &ncols, NULL);
    ocl_check(err, "get image width");

    err = clGetImageInfo(d_input, CL_IMAGE_HEIGHT, sizeof(nrows),
                         &nrows, NULL);
    ocl_check(err, "get image height");

    const size_t lws[] = { 32, 8 };
    const size_t gws[] = {
                           round_mul_up(ncols, lws[0]),
                           round_mul_up(nrows, lws[1]),
                         };

    err = clSetKernelArg(conv_k, arg++, sizeof(d_input), &d_input);
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

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        fprintf(stderr, "Usa la seguente sintassi: %s src_img dest_img filter\n"
                        "filter può assumere i seguenti valori:\n"
                        "- identity (filtro identità)\n"
                        "- Nbox3 (filtro di media 3x3)\n"
                        "- Nbox5 (filtro di media 5x5)\n"
                        "- sobelX (filtro di scansione dei dettagli orizzontali)\n"
                        "- sobelY (filtro di scansione dei dettagli verticali)\n"
                        "- laplace (filtro di edge detection)\n"
                        "- sharp (filtro di sharpening)\n", argv[0]);
        exit(1);
    }

    const char *src = argv[1];
    const char *dest = argv[2];
    const char *filter = argv[3];
    if (
        strcmp(filter, "identity") != 0 && strcmp(filter, "Nbox3") != 0 &&
        strcmp(filter, "Nbox5") != 0 && strcmp(filter, "sobelX") != 0 &&
        strcmp(filter, "sobelY") != 0 && strcmp(filter, "laplace") != 0 &&
        strcmp(filter, "sharp") != 0
       )
       error("Il filtro inserito non è valido, inserisci uno dei valori "
             "suggeriti dal prompt");

    imgInfo img;
    cl_int err;

    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p);
    cl_context ctx = create_context(p, d);
    cl_command_queue que = create_queue(ctx, d);
    cl_program prog = create_program("progetto.ocl", ctx, d);

    //Caricamento immagine
    err = load_pam(src, &img);
    if(err)
        error("caricamento immagine");
    if(img.channels < 3)
        error("supportiamo solo immagini a 3 o 4 canali");

    const size_t npixels = img.width*img.height;
    const size_t f_dim = strcmp(filter, "Nbox5") == 0 ? 5 : 3;
    const size_t memsize_ker = f_dim*f_dim*sizeof(real);

    if (img.width <= 0 || img.height <= 0)
        error("le dimensioni dell'immagine devono essere positive");

    //Creazione kernel
    cl_kernel kerinit_k = clCreateKernel(prog, filter, &err);
    ocl_check(err, "create kernel kerinit");

    cl_kernel conv_k = clCreateKernel(prog, "conv", &err);
    ocl_check(err, "create kernel conv");

    //Allocazione immagine
    cl_image_format fmt = {
        .image_channel_order = CL_RGBA,
        .image_channel_data_type = (img.depth == 8 ? CL_UNSIGNED_INT8 : CL_UNSIGNED_INT16)
    };

    cl_image_desc img_desc;
    memset(&img_desc, 0, sizeof(img_desc));
    img_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    img_desc.image_width = img.width;
    img_desc.image_height = img.height;

    //Immagine di input
    cl_mem d_src_img = clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &img_desc,
                                     NULL, &err);
    ocl_check(err, "create image src_img");
    //Immagine finale
    cl_mem d_dest_img = clCreateImage(ctx, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                                      &fmt, &img_desc, NULL, &err);
    ocl_check(err, "create image dest_img");
    //Buffers
    cl_mem d_ker = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                                  memsize_ker, NULL, &err);
    ocl_check(err, "create buffer ker");

    cl_event evt_upload, evt_download;
    const size_t origin[] = {0,0,0};
    const size_t region[] = {img.width, img.height, 1};

    //Load immagine
    err = clEnqueueWriteImage(que, d_src_img, CL_TRUE, origin, region, 0, 0,
                              img.data, 0, NULL, &evt_upload);
    ocl_check(err, "upload src_img");

    //Lancio kernel
    cl_event evt_init = kerinit(que, kerinit_k, d_ker, f_dim, f_dim, 0, NULL);

    cl_event evt_conv = conv(que, conv_k, d_src_img, d_ker, f_dim, d_dest_img,
                             1, &evt_init);

    //Download risultato
    memset(img.data, 0, img.data_size);
    err = clEnqueueReadImage(que, d_dest_img, CL_TRUE, origin, region, 0, 0,
                             img.data, 1, &evt_conv, &evt_download);
    ocl_check(err, "download su dest_img");

    err = clFinish(que);
    ocl_check(err, "clFinish");

    //Prestazioni
    printf("upload: %gms\t%gGB/s\n", runtime_ms(evt_upload),
           (1.0*img.data_size)/runtime_ns(evt_upload));

    printf("kerinit: %gms\t%GB/s\n", runtime_ms(evt_init),
           (1.0*memsize_ker)/runtime_ns(evt_init));

    printf("conv: %gms\t%gGB/s\n", runtime_ms(evt_conv),
           (10.0*img.data_size)/runtime_ns(evt_conv));

    printf("download: %gms\t%gGB/s\n", runtime_ms(evt_download),
           (1.0*img.data_size)/runtime_ns(evt_download));

    //Salvataggio immagine
    err = save_pam(dest, &img);
    if (err)
        error("impossibile salvare il file");


    clReleaseMemObject(d_src_img);
    clReleaseMemObject(d_dest_img);
    clReleaseMemObject(d_ker);

    clReleaseKernel(kerinit_k);
    clReleaseKernel(conv_k);
    clReleaseProgram(prog);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);
}
