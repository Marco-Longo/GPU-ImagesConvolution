#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "fft.h"

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
  printf("\nTrasformata (Spettro):\n");
  stampa(output, M, N);

  //ANTI-FFT
  double **res2 = anti_fft(res, M, N);
  printf("\nAnti-Trasformata:\n");
  stampa_d(res2, M, N);

  return 0;
}
