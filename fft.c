#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
//#include <complex.h>

typedef struct
{
  double real;
  double imag;
} complex;

void stampa(int **m, int M, int N)
{
  for(int i=0; i<M; i++)
  {
    for(int j=0; j<N; j++)
      printf("%d\t", m[i][j]);
    printf("\n");
  }
}

void stampa_complex(complex **m, int M, int N)
{
  for(int i=0; i<M; i++)
  {
    for(int j=0; j<N; j++)
      printf("%f + %fi\t", m[i][j].real, m[i][j].imag);
    printf("\n");
  }
}

int ** init(int M, int N)
{
  int **m = malloc(M * sizeof(int *));
  for(int i=0; i<M; i++)
  {
    m[i] = malloc(N * sizeof(int));
    for(int j=0; j<N; j++)
      //m[i][j] = rand() % 10;
      m[i][j] = 10 + 10*j + N*10*i;
  }

  return m;
}

int ** init0(int M, int N)
{
  int **m = malloc(M * sizeof(int *));
  for(int i=0; i<M; i++)
  {
    m[i] = malloc(N * sizeof(int));
    for(int j=0; j<N; j++)
      m[i][j] = 0;
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
    {
      m[i][j].real = 0;
      m[i][j].imag = 0;
    }
  }

  return m;
}

int ** fft_spectre(complex **F, int M, int N)
{
  int **m = init0(M, N);
  for(int i=0; i<M; i++)
    for(int j=0; j<N; j++)
      m[i][j] = (int)sqrt(pow(F[i][j].real, 2) + pow(F[i][j].imag, 2));

  return m;
}


/*****************************/
//Fast Fourier Transform (FFT)
complex ** fft(int **f, int M, int N)
{
  complex **F = init_complex(M,N);
  double pi = 4 * atan(1);
  int c = M*N;

  for(int u=0; u<M; u++)
    for(int v=0; v<N; v++)
    {
      for(int x=0; x<M; x++)
        for(int y=0; y<N; y++)
        {
          double r = 2 * pi * (((u*x)/(double)M) + ((v*y)/(double)N));
          //printf("%g\n", r);
          complex z = { cos(r), -sin(r) };
          //printf("%e + %e * i\n", z.real, z.imag);
          complex _z = { (f[x][y] * z.real), (f[x][y] * z.imag) };
          //printf("%e + %e * i\n", z.real, z.imag);
          F[u][v].real += _z.real;
          F[u][v].imag += _z.imag;
        }

      F[u][v].real /= c;
      F[u][v].imag /= c;
    }

  return F;
}
/*****************************/


int main(int argc, char **argv)
{
  if(argc != 3)
  {
    fprintf(stderr, "Use: %s rows cols\n", argv[0]);
    exit(1);
  }

  srand(time(NULL));
  int M = atoi(argv[1]);
  int N = atoi(argv[2]);

  int **input = init(M, N);
  printf("Iniziale:\n");
  stampa(input, M, N);

  //FFT
  complex **res = fft(input, M, N);
  int **output = fft_spectre(res, M, N);
  printf("\nTrasformata:\n");
  stampa_complex(res, M, N);
  //stampa(output, M, N);

  int somma = 0;
  for(int i=0; i<M; i++)
    for(int j=0; j<N; j++)
      somma += input[i][j];

  printf("\nValor medio: %d => %d\n", somma/(M*N), output[0][0]);

  /*
  double pi = 4 * atan(1);
  double complex z = cexp(I * pi);

  //e^(iπ) + 1 = 0
  printf("%f + %f * i\n", creal(z), cimag(z));
  z = 5 + 3 * I;
  printf("%f + %f * i\n", creal(z), cimag(z));
  */

  return 0;
}
