#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void stampa(float **m, int M, int N)
{
  for(int i=0; i<M; i++)
  {
    for(int j=0; j<N; j++)
      printf("%f\t", m[i][j]);
    printf("\n");
  }
}

float ** init(int M, int N)
{
  float **m = malloc(M * sizeof(float *));
  for(int i=0; i<M; i++)
  {
    m[i] = malloc(N * sizeof(float));
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


/************************/
//Convoluzione
float ** convoluzione(float **m, float **k, int M, int N)
{
  float **res = init(M, N);

  for(int i=0; i<M; i++)
    for(int j=0; j<N; j++)
    {
      int **tmp = init0(3, 3); //presuppone kernel di dimensione 3x3

      //Riempimento tmp
      tmp[1][1] = m[i][j]; //Elemento centrale (sempre presente)

      if(i == 0) //Elementi superiori inesistenti
        tmp[0][0] = tmp[0][1] = tmp[0][2] = 0;
      else
      {
        tmp[0][1] = m[i-1][j];
        if(j != N-1)  tmp[0][2] = m[i-1][j+1]; //
      }
      if(j == 0) //Elementi a sinistra inesistenti
        tmp[0][0] = tmp[1][0] = tmp[2][0] = 0;
      else
      {
        if(i != 0)  tmp[0][0] = m[i-1][j-1];
        tmp[1][0] = m[i][j-1];
        if(i != M-1)  tmp[2][0] = m[i+1][j-1]; //
      }

      if(i == M-1) //Elementi inferiori inesistenti
        tmp[2][0] = tmp[2][1] = tmp[2][2] = 0;
      else
      {
        tmp[2][1] = m[i+1][j];
        if(j != N-1)  tmp[2][2] = m[i+1][j+1];
      }
      if(j == N-1) //Elementi a destra inesistenti
        tmp[0][2] = tmp[1][2] = tmp[2][2] = 0;
      else
        tmp[1][2] = m[i][j+1];


      //printf("\nTmp %d, %d:\n", i, j);
      //stampa(tmp, 3, 3);

      //Esecuzione
      float acc = 0;
      for(int u=0; u<3; u++)
        for(int v=0; v<3; v++)
          acc += (tmp[u][v] * k[u][v]);

      res[i][j] = acc;
    }

  return res;
}
/************************/


int main(int argc, char **argv)
{
  if(argc != 3)
  {
    fprintf(stderr, "Use: %s rows cols\n", argv[0]);
    exit(1);
  }

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  float **kernel = init_kernel(3, 3);

  float **img = init(M, N);
  printf("Iniziale:\n");
  stampa(img, M, N);

  float **res = convoluzione(img, kernel, M, N);
  printf("\nTrasformata:\n");
  stampa(res, M, N);

  return 0;
}
