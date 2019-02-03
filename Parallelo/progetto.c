#define CL_TARGET_OPENCL_VERSION 120
#include "ocl_boiler.h"

#include <stdio.h>
#include <stdlib.h>
#include "pamalign.h"

typedef struct
{
  double real;
  double imag;
} complex;

void stampa_complex(complex *m, int M, int N)
{
  for(int i=0; i<5; i++)
  {
    for(int j=0; j<5; j++)
      printf("%f + %fi\t", m[i*N+j].real, m[i*N+j].imag);
    printf("\n");
  }
}

void error(const char *msg)
{
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

//Funzione per lancio del kernel
cl_event fft(cl_command_queue que, cl_kernel fft_k, cl_mem d_src_img,
             cl_mem d_dest_mat, int u, int v, int n_wait_events, cl_event *wait_events)
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

    err = clSetKernelArg(fft_k, arg++, sizeof(d_src_img), &d_src_img);
    ocl_check(err, "set fft arg %d", arg - 1);
    err = clSetKernelArg(fft_k, arg++, sizeof(d_dest_mat), &d_dest_mat);
    ocl_check(err, "set fft arg %d", arg - 1);
    err = clSetKernelArg(fft_k, arg++, sizeof(complex), NULL);
    ocl_check(err, "set fft arg %d", arg - 1);
    err = clSetKernelArg(fft_k, arg++, sizeof(u), &u);
    ocl_check(err, "set fft arg %d", arg - 1);
    err = clSetKernelArg(fft_k, arg++, sizeof(v), &v);
    ocl_check(err, "set fft arg %d", arg - 1);

    const size_t lws[] = {16, 16};
    const size_t gws[] = {
        round_mul_up(ncols, lws[0]),
        round_mul_up(nrows, lws[1]),
    };

    cl_event evt_fft;
    err = clEnqueueNDRangeKernel(que, fft_k, 2, NULL, gws, lws, n_wait_events,
                                 wait_events, &evt_fft);
    ocl_check(err, "enqueue fft");

    return evt_fft;
}

/*Verifica dei risultati
void verify(const int *img, int nrows, int ncols, size_t pitch)
{
    size_t pitch_el = pitch/sizeof(*img);
    if (pitch != pitch_el*sizeof(*img))
        fprintf(stderr, "pitch non allineato: %zu / %zu = %zu", pitch, sizeof(*img), pitch_el);

    for (int r = 0; r < nrows; ++r) {
        for (int c = 0; c < ncols; ++c) {
            int atteso = 255;
            int reale = img[r*pitch_el + c];
            if (atteso != reale) {
                fprintf(stderr, "mismatch @ %d %d: %d != %d\n", r, c, reale, atteso);
                exit(2);
            }
        }
    }
}*/

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "%s src dest\n", argv[0]);
        exit(1);
    }

    const char *src = argv[1];
    const char *dest = argv[2];
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
    if(img.channels != 3)
        error("supportiamo solo immagini a 3 canali");

    size_t npixels = img.width*img.height;

    //Creazione kernel
    cl_kernel fft_k = clCreateKernel(prog, "fft", &err);
    ocl_check(err, "create kernel fft");

    //Allocazione immagine
    cl_image_format fmt = {
        .image_channel_order = CL_RGBA,
        .image_channel_data_type = (img.depth == 8 ? CL_SIGNED_INT8 : CL_SIGNED_INT16)
    };

    cl_image_desc img_desc;
    memset(&img_desc, 0, sizeof(img_desc));

    img_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    img_desc.image_width = img.width;
    img_desc.image_height = img.height;

    //Immagine di input
    cl_mem d_src_img = clCreateImage(ctx, CL_MEM_READ_WRITE, &fmt, &img_desc,
                                     NULL, &err);
    ocl_check(err, "create image src_img");
    //Immagine finale
    cl_mem d_dest_img = clCreateImage(ctx, CL_MEM_WRITE_ONLY |
      CL_MEM_HOST_READ_ONLY, &fmt, &img_desc, NULL, &err);
    ocl_check(err, "create image dest_img");
    //Buffer intermedio per fft
    size_t memsize_cplx = npixels*sizeof(complex);
    cl_mem d_vfft = clCreateBuffer(ctx, CL_MEM_READ_WRITE |
      CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, memsize_cplx, NULL, &err);

    cl_event evt_upload, evt_download;
    cl_event* evt_fft = malloc(npixels*sizeof(cl_event));

    const size_t origin[] = {0,0,0};
    const size_t region[] = {img.width, img.height, 1};

    //Load immagine
    err = clEnqueueWriteImage(que, d_src_img, CL_TRUE, origin, region, 0, 0,
      img.data, 0, NULL, &evt_upload);
    ocl_check(err, "upload src_img");

    //Lancio kernel
    for(int u=0; u<img.height; ++u)
      for(int v=0; v<img.width; ++v)
        evt_fft[u*img.width+v] = fft(que, fft_k, d_src_img, d_vfft, u, v,
                                      1, &evt_upload);

    /*Download risultato
    memset(img.data, 0, img.data_size);

    err = clEnqueueReadImage(que, d_dest_img, CL_TRUE, origin, region, 0, 0,
      img.data, 1, &evt_draw, &evt_download);
    ocl_check(err, "download su dest_img");
    */
    err = clFinish(que);
    ocl_check(err, "clFinish");

    //Prestazioni
    printf("upload: %gms\t%gGB/s\n", runtime_ms(evt_upload), (1.0*img.data_size)/runtime_ns(evt_upload));

    printf("fft: %gms\t%gGB/s\n", total_runtime_ms(evt_fft[0], evt_fft[npixels-1]),
      (img.data_size*(1.0+img.data_size))/total_runtime_ns(evt_fft[0], evt_fft[npixels-1]));

    //printf("download: %gms\t%gGB/s\n", runtime_ms(evt_download), (1.0*img.data_size)/runtime_ns(evt_download));

    /*Salvataggio immagine
    err = save_pam(dest, &img);
    if (err)
        error("impossibile salvare il file");
    */

    //Estrazione matrice complessa
    cl_event evt_map, evt_unmap;
    complex *h_trasf = clEnqueueMapBuffer(que, d_vfft, CL_TRUE, CL_MAP_READ,
      0, memsize_cplx, 0, NULL, &evt_map, &err);
    ocl_check(err, "map buffer vfft");
    printf("map: %gms\t%gGB/s\n", runtime_ms(evt_map),
            (1.0*memsize_cplx)/runtime_ns(evt_map));
    stampa_complex(h_trasf, img.height, img.width);
    err = clEnqueueUnmapMemObject(que, d_vfft, h_trasf,
      0, NULL, &evt_unmap);
    ocl_check(err, "unmap buffer vfft");
    clFinish(que);
    printf("unmap: %gms\t%gGB/s\n", runtime_ms(evt_unmap),
            (1.0*memsize_cplx)/runtime_ns(evt_unmap));


    clReleaseMemObject(d_src_img);
    clReleaseMemObject(d_dest_img);
    clReleaseMemObject(d_vfft);

    clReleaseKernel(fft_k);
    clReleaseProgram(prog);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);
}
