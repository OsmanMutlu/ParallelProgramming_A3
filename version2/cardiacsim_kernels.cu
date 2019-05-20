/*

	Implement your CUDA kernel in this file

*/

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
  int row = blockIdx.y*blockDim.y + threadIdx.y + 1;
  int col = blockIdx.x*blockDim.x + threadIdx.x + 1;

  if ((row - 1 < m) && (col - 1 < n)) {
    // We can't merge this kernel here. We need a global sync across blocks for that.
    // E_prev[row*(n+2)] = E_prev[row*(n+2) + 2];
    // E_prev[row*(n+2) + n + 1] = E_prev[row*(n+2) + n - 1];
    // E_prev[col] = E_prev[2*(n+2) + col];
    // E_prev[(m+1)*(n+2) + col] = E_prev[(m-1)*(n+2) + col];

    E[row*(n+2)+col] = E_prev[row*(n+2)+col] + alpha*(E_prev[row*(n+2)+col+1] + E_prev[row*(n+2)+col-1] - 4*E_prev[row*(n+2)+col] + E_prev[(row+1)*(n+2)+col] + E_prev[(row-1)*(n+2)+col]);
    E[row*(n+2)+col] = E[row*(n+2)+col] - dt*(kk*E[row*(n+2)+col]*(E[row*(n+2)+col] - a)*(E[row*(n+2)+col] - 1) + E[row*(n+2)+col]*R[row*(n+2)+col]);
    R[row*(n+2)+col] = R[row*(n+2)+col] + dt*(epsilon + M1*R[row*(n+2)+col]/(E[row*(n+2)+col] + M2))*(-R[row*(n+2)+col] - kk*E[row*(n+2)+col]*(E[row*(n+2)+col] - b - 1));
  }
}
