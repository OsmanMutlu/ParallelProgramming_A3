/*

	Implement your CUDA kernel in this file

*/

#define TILE_DIM 32

__global__ void mirror_boundaries(double *E_prev, const int n, const int m)
{
  int row = blockIdx.y*blockDim.y + threadIdx.y + 1;
  int col = blockIdx.x*blockDim.x + threadIdx.x + 1;

  if (col == 1) {
    E_prev[row*(n+2)] = E_prev[row*(n+2) + 2];
    E_prev[row*(n+2) + n + 1] = E_prev[row*(n+2) + n - 1];
  }
  if (row == 1) {
    E_prev[col] = E_prev[2*(n+2) + col];
    E_prev[(m+1)*(n+2) + col] = E_prev[(m-1)*(n+2) + col];
  }
}

__global__ void simulate (double *E, double *E_prev, double *R, const double alpha,
			  const int n, const int m, const double kk,
			  const double dt, const double a, const double epsilon,
			  const double M1,const double  M2, const double b)
{

  // __shared__ double E_Block[TILE_DIM][TILE_DIM];
  // __shared__ double R_Block[TILE_DIM][TILE_DIM];

  // int ty = threadIdx.y;
  // int tx = threadIdx.x;

  // int row = blockIdx.y*blockDim.y + threadIdx.y + 1;
  // int col = blockIdx.x*blockDim.x + threadIdx.x + 1;

  // if ((row - 1 < m) && (col - 1 < n)) {

  //   E_Block[ty][tx] = E[row*(n+2) + col];
  //   R_Block[ty][tx] = R[row*(n+2) + col];

  //   E[row*(n+2)+col] = E_prev[row*(n+2)+col] + alpha*(E_prev[row*(n+2)+col+1] + E_prev[row*(n+2)+col-1] - 4*E_prev[row*(n+2)+col] + E_prev[(row+1)*(n+2)+col] + E_prev[(row-1)*(n+2)+col]);

  //   E[row*(n+2)+col] = E_Block[ty][tx] = E_Block[ty][tx] - dt*(kk*E_Block[ty][tx]*(E_Block[ty][tx] - a)*(E_Block[ty][tx] - 1) + E_Block[ty][tx]*R_Block[ty][tx]);
  //   R[row*(n+2)+col] = R_Block[ty][tx] + dt*(epsilon + M1*R_Block[ty][tx]/(E_Block[ty][tx] + M2))*(-R_Block[ty][tx] - kk*E_Block[ty][tx]*(E_Block[ty][tx] - b - 1));
  // }



  __shared__ double E_Block[TILE_DIM][TILE_DIM];
  __shared__ double R_Block[TILE_DIM][TILE_DIM];
  __shared__ double E_prev_Block[TILE_DIM+2][TILE_DIM+2];

  int ty = threadIdx.y;
  int tx = threadIdx.x;

  int row = blockIdx.y*blockDim.y + threadIdx.y + 1;
  int col = blockIdx.x*blockDim.x + threadIdx.x + 1;

  if ((row - 1 < m) && (col - 1 < n)) {
    // E_prev_Block[ty][tx] = E_prev[row*(n+2) + col];

    // if (ty == 2) {
    //   E_prev_Block[0][tx] = E_prev[(row-2)*(n+2) + col];
    //   E_prev_Block[TILE_DIM+1][tx] = E_prev[(row+TILE_DIM-1)*(n+2) + col];
    // }

    // if (tx == 2) {
    //   E_prev_Block[ty][0] = E_prev[row*(n+2) + col-2];
    //   E_prev_Block[ty][TILE_DIM+1] = E_prev[row*(n+2) + col+TILE_DIM-1];
    // }

    // E[row*(n+2)+col] = E_prev_Block[ty][tx] + alpha*(E_prev_Block[ty][tx+1] + E_prev_Block[ty][tx-1] - 4*E_prev_Block[ty][tx] + E_prev_Block[ty+1][tx] + E_prev_Block[ty-1][tx]);


    E_prev_Block[ty+1][tx+1] = E_prev[row*(n+2) + col];

    if (ty == 1) {
      E_prev_Block[0][tx+1] = E_prev[(row-2)*(n+2) + col];
      E_prev_Block[TILE_DIM+1][tx+1] = E_prev[(row+TILE_DIM-1)*(n+2) + col];
    }

    if (tx == 1) {
      E_prev_Block[ty+1][0] = E_prev[row*(n+2) + col-2];
      E_prev_Block[ty+1][TILE_DIM+1] = E_prev[row*(n+2) + col+TILE_DIM-1];
    }

    E[row*(n+2)+col] = E_prev_Block[ty+1][tx+1] + alpha*(E_prev_Block[ty+1][tx+2] + E_prev_Block[ty+1][tx] - 4*E_prev_Block[ty+1][tx+1] + E_prev_Block[ty+2][tx+1] + E_prev_Block[ty][tx+1]);


    // E[row*(n+2)+col] = E_prev[row*(n+2)+col] + alpha*(E_prev[row*(n+2)+col+1] + E_prev[row*(n+2)+col-1] - 4*E_prev[row*(n+2)+col] + E_prev[(row+1)*(n+2)+col] + E_prev[(row-1)*(n+2)+col]);

    E_Block[ty][tx] = E[row*(n+2) + col];
    R_Block[ty][tx] = R[row*(n+2) + col];

    E[row*(n+2)+col] = E_Block[ty][tx] = E_Block[ty][tx] - dt*(kk*E_Block[ty][tx]*(E_Block[ty][tx] - a)*(E_Block[ty][tx] - 1) + E_Block[ty][tx]*R_Block[ty][tx]);
    R[row*(n+2)+col] = R_Block[ty][tx] + dt*(epsilon + M1*R_Block[ty][tx]/(E_Block[ty][tx] + M2))*(-R_Block[ty][tx] - kk*E_Block[ty][tx]*(E_Block[ty][tx] - b - 1));
  }

}
