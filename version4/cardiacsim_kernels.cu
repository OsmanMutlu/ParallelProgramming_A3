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

  __shared__ double E_Block[TILE_DIM][TILE_DIM];
  __shared__ double R_Block[TILE_DIM][TILE_DIM];

  int ty = threadIdx.y;
  int tx = threadIdx.x;

  int row = blockIdx.y*blockDim.y + threadIdx.y + 1;
  int col = blockIdx.x*blockDim.x + threadIdx.x + 1;

  if ((row - 1 < m) && (col - 1 < n)) {

    E[row*(n+2)+col] = E_prev[row*(n+2)+col] + alpha*(E_prev[row*(n+2)+col+1] + E_prev[row*(n+2)+col-1] - 4*E_prev[row*(n+2)+col] + E_prev[(row+1)*(n+2)+col] + E_prev[(row-1)*(n+2)+col]);

    E_Block[ty][tx] = E[row*(n+2) + col];
    R_Block[ty][tx] = R[row*(n+2) + col];

    E[row*(n+2)+col] = E_Block[ty][tx] = E_Block[ty][tx] - dt*(kk*E_Block[ty][tx]*(E_Block[ty][tx] - a)*(E_Block[ty][tx] - 1) + E_Block[ty][tx]*R_Block[ty][tx]);
    R[row*(n+2)+col] = R_Block[ty][tx] + dt*(epsilon + M1*R_Block[ty][tx]/(E_Block[ty][tx] + M2))*(-R_Block[ty][tx] - kk*E_Block[ty][tx]*(E_Block[ty][tx] - b - 1));
  }

}
