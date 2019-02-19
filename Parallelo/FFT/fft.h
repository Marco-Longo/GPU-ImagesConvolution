#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define _2PI 6.283185307179586476925f
typedef double ctype;

typedef struct
{
  ctype real;
  ctype imag;
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

void stampa_f(float **m, int M, int N)
{
  for(int i=0; i<M; i++)
  {
    for(int j=0; j<N; j++)
      printf("%e\t", m[i][j]);
    printf("\n");
  }
}

void stampa_d(double **m, int M, int N)
{
  for(int i=0; i<M; i++)
  {
    for(int j=0; j<N; j++)
      printf("%g\t", m[i][j]);
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
      m[i][j] = i+j;
      //m[i][j] = 10 + 10*j + N*10*i;
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

float ** init0_f(int M, int N)
{
  float **m = malloc(M * sizeof(float *));
  for(int i=0; i<M; i++)
  {
    m[i] = malloc(N * sizeof(float));
    for(int j=0; j<N; j++)
      m[i][j] = 0.0;
  }

  return m;
}

double ** init0_d(int M, int N)
{
  double **m = malloc(M * sizeof(double *));
  for(int i=0; i<M; i++)
  {
    m[i] = malloc(N * sizeof(double));
    for(int j=0; j<N; j++)
      m[i][j] = 0.0;
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

float ** init_kernel(int M, int N)
{
  float **k = malloc(M * sizeof(float *));
  for(int i=0; i<M; i++)
  {
    k[i] = malloc(N * sizeof(float));
    for(int j=0; j<N; j++)
      k[i][j] = 1.0/(M*N);
  }

  return k;
}

int ** fft_spectre(complex **F, int M, int N)
{
  int **m = init0(M, N);
  for(int i=0; i<M; i++)
    for(int j=0; j<N; j++)
      m[i][j] = (int)sqrt(pow(F[i][j].real, 2) + pow(F[i][j].imag, 2));

  return m;
}

float ** fft_spectre_f(complex **F, int M, int N)
{
  float **m = init0_f(M, N);
  for(int i=0; i<M; i++)
    for(int j=0; j<N; j++)
      m[i][j] = (float)sqrt(pow(F[i][j].real, 2) + pow(F[i][j].imag, 2));

  return m;
}


/*****************************/
//Fast Fourier Transform (FFT)
complex ** fft_c(int **f, int M, int N)
{
  complex **F = init_complex(M,N);
  //double pi = 4 * atan(1);
  //int c = M*N;

  for(int u=0; u<M; u++)
    for(int v=0; v<N; v++)
    {
      for(int x=0; x<M; x++)
        for(int y=0; y<N; y++)
        {
          ctype r = _2PI * (((u*x)/(ctype)M) + ((v*y)/(ctype)N));
          //printf("%g\n", r);
          complex z = { cos(r), -sin(r) };
          //printf("%e + %e * i\n", z.real, z.imag);
          complex _z = { (f[x][y] * z.real), (f[x][y] * z.imag) };
          //printf("%e + %e * i\n", z.real, z.imag);
          F[u][v].real += _z.real;
          F[u][v].imag += _z.imag;
        }

      //F[u][v].real /= c;
      //F[u][v].imag /= c;
    }

  return F;
}
complex ** fft_f(float **f, int M, int N)
{
  complex **F = init_complex(M,N);
  //double pi = 4 * atan(1);
  //int c = M*N;

  for(int u=0; u<M; u++)
    for(int v=0; v<N; v++)
    {
      for(int x=0; x<M; x++)
        for(int y=0; y<N; y++)
        {
          ctype r = _2PI * (((u*x)/(ctype)M) + ((v*y)/(ctype)N));
          //printf("%g\n", r);
          complex z = { cos(r), -sin(r) };
          //printf("%e + %e * i\n", z.real, z.imag);
          complex _z = { (f[x][y] * z.real), (f[x][y] * z.imag) };
          //printf("%e + %e * i\n", z.real, z.imag);
          F[u][v].real += _z.real;
          F[u][v].imag += _z.imag;
        }

      //F[u][v].real /= c;
      //F[u][v].imag /= c;
    }

  return F;
}
/*****************************/

/*****************************/
//Fast Fourier Anti-Transform (FFT)
float ** anti_fft(complex **F, int M, int N)
{
  float **f = init0_f(M,N);
  //double pi = 4 * atan(1);
  ctype c = M*N*1.0f;

  for(int x=0; x<M; x++)
    for(int y=0; y<N; y++)
    {
      for(int u=0; u<M; u++)
        for(int v=0; v<N; v++)
        {
          ctype r = _2PI * (((u*x)/(ctype)M) + ((v*y)/(ctype)N));

          complex z = { cos(r), sin(r) };
          //printf("%e + %e * i\n", z.real, z.imag);
          complex _z = { (F[u][v].real * z.real)-(F[u][v].imag * z.imag),
                         (F[u][v].real * z.imag)+(F[u][v].imag * z.real) };

          //printf("\nParte immaginaria: %e", _z.imag);
          //f[x][y] += sqrt(pow(_z.real, 2) + pow(_z.imag, 2));
          f[x][y] += _z.real;
          //if(u==M-1 && v==N-1)  printf("\nParte immaginaria: %e", _z.imag);
        }

      f[x][y] /= c;
    }

  return f;
}
/*****************************/
