#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"
#include "pamalign.h"
#include <stdio.h>
#include <stdlib.h>

typedef double     real;
typedef cl_double2 complex;

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

cl_event fft_mat(cl_command_queue que, cl_kernel fft_mat_k, cl_mem d_src_mat,
                 cl_mem d_dest_mat, int nrows, int ncols,
                 int n_wait_events, cl_event *wait_events)
{
    cl_int err;
    cl_int arg = 0;

    err = clSetKernelArg(fft_mat_k, arg++, sizeof(d_src_mat), &d_src_mat);
    ocl_check(err, "set fft_mat arg %d", arg - 1);
    err = clSetKernelArg(fft_mat_k, arg++, sizeof(d_dest_mat), &d_dest_mat);
    ocl_check(err, "set fft_mat arg %d", arg - 1);
    err = clSetKernelArg(fft_mat_k, arg++, sizeof(nrows), &nrows);
    ocl_check(err, "set fft_mat arg %d", arg - 1);
    err = clSetKernelArg(fft_mat_k, arg++, sizeof(ncols), &ncols);
    ocl_check(err, "set fft_mat arg %d", arg - 1);

    const size_t lws[] = {16, 16};
    const size_t gws[] = {
                           round_mul_up(nrows, lws[0]),
                           round_mul_up(ncols, lws[1]),
                         };

    cl_event evt_fft;
    err = clEnqueueNDRangeKernel(que, fft_mat_k, 2, NULL, gws, lws, n_wait_events,
                                 wait_events, &evt_fft);
    ocl_check(err, "enqueue fft_mat");

    return evt_fft;
}

cl_event fft_img(cl_command_queue que, cl_kernel fft_img_k, cl_mem d_src_img,
                 cl_mem d_dest_mat, int n_wait_events, cl_event *wait_events)
{
    cl_int err;
    cl_int arg = 0;
    size_t ncols, nrows;

    err = clGetImageInfo(d_src_img, CL_IMAGE_WIDTH, sizeof(ncols),
                         &ncols, NULL);
    ocl_check(err, "get image width");

    err = clGetImageInfo(d_src_img, CL_IMAGE_HEIGHT, sizeof(nrows),
                         &nrows, NULL);
    ocl_check(err, "get image height");

    err = clSetKernelArg(fft_img_k, arg++, sizeof(d_src_img), &d_src_img);
    ocl_check(err, "set fft_img arg %d", arg - 1);
    err = clSetKernelArg(fft_img_k, arg++, sizeof(d_dest_mat), &d_dest_mat);
    ocl_check(err, "set fft_img arg %d", arg - 1);

    const size_t lws[] = {16, 16};
    const size_t gws[] = {
                           round_mul_up(nrows, lws[0]),
                           round_mul_up(ncols, lws[1]),
                         };

    cl_event evt_fft;
    err = clEnqueueNDRangeKernel(que, fft_img_k, 2, NULL, gws, lws, n_wait_events,
                                 wait_events, &evt_fft);
    ocl_check(err, "enqueue fft");

    return evt_fft;
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

cl_event ifft(cl_command_queue que, cl_kernel ifft_k, cl_mem d_src_mat,
              cl_mem d_dest_img, int n_wait_events, cl_event *wait_events)
{
    cl_int err;
    cl_int arg = 0;
    size_t ncols, nrows;

    err = clGetImageInfo(d_dest_img, CL_IMAGE_WIDTH, sizeof(ncols),
                         &ncols, NULL);
    ocl_check(err, "get image width");

    err = clGetImageInfo(d_dest_img, CL_IMAGE_HEIGHT, sizeof(nrows),
                         &nrows, NULL);
    ocl_check(err, "get image height");

    err = clSetKernelArg(ifft_k, arg++, sizeof(d_src_mat), &d_src_mat);
    ocl_check(err, "set ifft arg %d", arg - 1);
    err = clSetKernelArg(ifft_k, arg++, sizeof(d_dest_img), &d_dest_img);
    ocl_check(err, "set ifft arg %d", arg - 1);

    const size_t lws[] = {16, 16};
    const size_t gws[] = {
                           round_mul_up(nrows, lws[0]),
                           round_mul_up(ncols, lws[1]),
                         };

    cl_event evt_ifft;
    err = clEnqueueNDRangeKernel(que, ifft_k, 2, NULL, gws, lws, n_wait_events,
                                 wait_events, &evt_ifft);
    ocl_check(err, "enqueue ifft");

    return evt_ifft;
}


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
    int len = sizeof(filter);

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

    size_t npixels = img.width*img.height;
    size_t memsize_real = npixels*sizeof(real);
    size_t memsize_complex = npixels*sizeof(complex);

    if (img.width <= 0 || img.height <= 0)
        error("le dimensioni dell'immagine devono essere positive");
    if (npixels & (npixels-1))
        error("il numero di pixels deve essere una potenza di due");

    //Creazione kernel
    cl_kernel kerinit_k = clCreateKernel(prog, filter, &err);
    ocl_check(err, "create kernel kerinit");

    cl_kernel fft_mat_k = clCreateKernel(prog, "fft_mat", &err);
    ocl_check(err, "create kernel fft_mat");

    cl_kernel fft_img_k = clCreateKernel(prog, "fft_img", &err);
    ocl_check(err, "create kernel fft_img");

    cl_kernel prod_k = clCreateKernel(prog, "product", &err);
    ocl_check(err, "create kernel prod");

    cl_kernel ifft_k = clCreateKernel(prog, "ifft", &err);
    ocl_check(err, "create kernel ifft");

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
    cl_mem d_ker1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                                   memsize_real, NULL, &err);
    ocl_check(err, "create buffer ker1");
    cl_mem d_vfft = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                                   memsize_complex, NULL, &err);
    ocl_check(err, "create buffer vfft");
    cl_mem d_ker2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                                   memsize_complex, NULL, &err);
    ocl_check(err, "create buffer ker2");
    cl_mem d_vprod = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                                    memsize_complex, NULL, &err);
    ocl_check(err, "create buffer vprod");

    cl_event evt_upload, evt_download;
    const size_t origin[] = {0,0,0};
    const size_t region[] = {img.width, img.height, 1};

    //Load immagine
    err = clEnqueueWriteImage(que, d_src_img, CL_TRUE, origin, region, 0, 0,
                              img.data, 0, NULL, &evt_upload);
    ocl_check(err, "upload src_img");

    //Lancio kernel
    cl_event evt_init = kerinit(que, kerinit_k, d_ker1, img.height, img.width,
                                0, NULL);

    cl_event evt_fft[2];
    evt_fft[0] = fft_img(que, fft_img_k, d_src_img, d_vfft, 1, &evt_upload);
    evt_fft[1] = fft_mat(que, fft_mat_k, d_ker1, d_ker2, img.height, img.width,
                         1, &evt_init);

    cl_event evt_prod = product(que, prod_k, d_vfft, d_ker2, d_vprod, img.height,
                                img.width, 2, evt_fft);

    cl_event evt_ifft = ifft(que, ifft_k, d_vprod, d_dest_img, 1, &evt_prod);

    //Download risultato
    memset(img.data, 0, img.data_size);
    err = clEnqueueReadImage(que, d_dest_img, CL_TRUE, origin, region, 0, 0,
                             img.data, 1, &evt_ifft, &evt_download);
    ocl_check(err, "download su dest_img");

    err = clFinish(que);
    ocl_check(err, "clFinish");

    //Prestazioni
    printf("upload: %gms\t%gGB/s\n", runtime_ms(evt_upload),
           (1.0*img.data_size)/runtime_ns(evt_upload));

    printf("download: %gms\t%gGB/s\n", runtime_ms(evt_download),
           (1.0*img.data_size)/runtime_ns(evt_download));

    //Salvataggio immagine
    err = save_pam(dest, &img);
    if (err)
        error("impossibile salvare il file");


    clReleaseMemObject(d_src_img);
    clReleaseMemObject(d_dest_img);
    clReleaseMemObject(d_vfft);
    clReleaseMemObject(d_ker1);
    clReleaseMemObject(d_ker2);
    clReleaseMemObject(d_vprod);

    clReleaseKernel(kerinit_k);
    clReleaseKernel(fft_mat_k);
    clReleaseKernel(fft_img_k);
    clReleaseKernel(ifft_k);
    clReleaseKernel(prod_k);
    clReleaseProgram(prog);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);
}
