#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "fft.h"

float ** adapt_kernel(float **k, int M_i, int N_i, //dimensioni del kernel
                                 int M_f, int N_f) //dimensioni dell'immagine
{
  float **m = init0_f(M_f, N_f);

  for(int i=0; i<M_i; i++)
    for(int j=0; j<N_i; j++)
      m[i][j] = k[i][j];

  return m;
}

complex ** conv(complex **m, complex **k, int M, int N)
{
  complex **res = init_complex(M, N);

  for(int i=0; i<M; i++)
    for(int j=0; j<N; j++)
    {
      double a = m[i][j].real;
      double b = m[i][j].imag;
      double c = k[i][j].real;
      double d = k[i][j].imag;

      res[i][j].real = a*c - b*d;
      res[i][j].imag = a*d + b*c;
    }

  return res;
}

float ** conv_spectre(float **m, float **k, int M, int N)
{
  float **res = init0_f(M, N);

  for(int i=0; i<M; i++)
    for(int j=0; j<N; j++)
      res[i][j] = m[i][j] * k[i][j];

  return res;
}


int main(int argc, char **argv)
{
  if(argc != 3)
  {
    fprintf(stderr, "Use: %s rows cols\n", argv[0]);
    exit(1);
  }

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  float **kernel = adapt_kernel(init_kernel(3, 3), 3, 3, M, N);
  /*
    //Kernel
    printf("\nKernel:\n");
    stampa_f(kernel, M, N);
  */
  int **img = init(M, N);
  printf("Iniziale:\n");
  stampa(img, M, N);
  //FFT
  complex **t_img = fft(img, M, N);
  float **s_img = fft_spectre_f(t_img, M, N);
  /*
  printf("\nTrasformata:\n");
  stampa_complex(t_img, M, N);
  printf("\nTrasformata (Spettro):\n");
  stampa_f(s_img, M, N);
  */
  complex **t_kernel = fft_f(kernel, M, N);
  float **s_kernel = fft_spectre_f(t_kernel, M, N);

  complex ** res = conv(t_img, t_kernel, M, N);

  printf("\nConvoluzione:\n");
  //stampa_complex(res, M, N);
  float **s_res = fft_spectre_f(res, M, N);
  stampa_f(s_res, M, N);

  printf("\nConvoluzione su spettro:\n");
  float **s_res2 = conv_spectre(s_img, s_kernel, M, N);
  stampa_f(s_res2, M, N);
}
