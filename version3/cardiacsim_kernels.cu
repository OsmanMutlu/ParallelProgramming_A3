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

    E[row*(n+2)+col] = E_prev[row*(n+2)+col] + alpha*(E_prev[row*(n+2)+col+1] + E_prev[row*(n+2)+col-1] - 4*E_prev[row*(n+2)+col] + E_prev[(row+1)*(n+2)+col] + E_prev[(row-1)*(n+2)+col]);

    double tmp_E = E[row*(n+2)+col];
    double tmp_R = R[row*(n+2)+col];

    E[row*(n+2)+col] = tmp_E = tmp_E - dt*(kk*tmp_E*(tmp_E - a)*(tmp_E - 1) + tmp_E*tmp_R);
    R[row*(n+2)+col] = tmp_R + dt*(epsilon + M1*tmp_R/(tmp_E + M2))*(-tmp_R - kk*tmp_E*(tmp_E - b - 1));
  }
}
