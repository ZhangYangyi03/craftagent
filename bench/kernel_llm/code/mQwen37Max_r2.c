#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void solve(double *K, double *F, double *U, int n) {
    int *piv = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) piv[i] = i;
    
    for (int k = 0; k < n; k++) {
        double max_val = fabs(K[piv[k] * n + k]);
        int max_idx = k;
        for (int i = k + 1; i < n; i++) {
            double val = fabs(K[piv[i] * n + k]);
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        if (max_idx != k) {
            int tmp = piv[k];
            piv[k] = piv[max_idx];
            piv[max_idx] = tmp;
        }
        
        int pk = piv[k];
        double diag = K[pk * n + k];
        
        for (int i = k + 1; i < n; i++) {
            int pi = piv[i];
            double factor = K[pi * n + k] / diag;
            K[pi * n + k] = factor;
            for (int j = k + 1; j < n; j++) {
                K[pi * n + j] -= factor * K[pk * n + j];
            }
            F[pi] -= factor * F[pk];
        }
    }
    
    for (int i = n - 1; i >= 0; i--) {
        int pi = piv[i];
        double sum = F[pi];
        for (int j = i + 1; j < n; j++) {
            sum -= K[pi * n + j] * U[j];
        }
        U[i] = sum / K[pi * n + i];
    }
    
    free(piv);
}

int main(int argc, char **argv) {
    if (argc < 6) return 1;
    
    double L = atof(argv[1]);
    double H = atof(argv[2]);
    int nx = atoi(argv[3]);
    int ny = atoi(argv[4]);
    double total_y = atof(argv[5]);
    
    int nnx = nx + 1;
    int nny = ny + 1;
    int N = nnx * nny;
    int ndof = 2 * N;
    
    double E = 210000.0;
    double nu = 0.3;
    double c = E / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double D[3][3] = {
        {c * (1.0 - nu), c * nu, 0.0},
        {c * nu, c * (1.0 - nu), 0.0},
        {0.0, 0.0, c * (1.0 - 2.0 * nu) / 2.0}
    };
    
    double J11 = L / (2.0 * nx);
    double J22 = H / (2.0 * ny);
    double detJ = J11 * J22;
    
    double Ke[8][8] = {0};
    double gp[2] = {-1.0 / sqrt(3.0), 1.0 / sqrt(3.0)};
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            double xi = gp[i];
            double eta = gp[j];
            double dN_dxi[4] = {
                -0.25 * (1.0 - eta), 0.25 * (1.0 - eta),
                 0.25 * (1.0 + eta), -0.25 * (1.0 + eta)
            };
            double dN_deta[4] = {
                -0.25 * (1.0 - xi), -0.25 * (1.0 + xi),
                 0.25 * (1.0 + xi),  0.25 * (1.0 - xi)
            };
            double dN_dx[4], dN_dy[4];
            for (int k = 0; k < 4; k++) {
                dN_dx[k] = dN_dxi[k] / J11;
                dN_dy[k] = dN_deta[k] / J22;
            }
            
            double B[3][8] = {0};
            for (int k = 0; k < 4; k++) {
                B[0][2 * k] = dN_dx[k];
                B[1][2 * k + 1] = dN_dy[k];
                B[2][2 * k] = dN_dy[k];
                B[2][2 * k + 1] = dN_dx[k];
            }
            
            double DB[3][8] = {0};
            for (int r = 0; r < 3; r++) {
                for (int cc = 0; cc < 8; cc++) {
                    for (int m = 0; m < 3; m++) {
                        DB[r][cc] += D[r][m] * B[m][cc];
                    }
                }
            }
            
            for (int r = 0; r < 8; r++) {
                for (int cc = 0; cc < 8; cc++) {
                    for (int m = 0; m < 3; m++) {
                        Ke[r][cc] += B[m][r] * DB[m][cc] * detJ;
                    }
                }
            }
        }
    }
    
    double *K = (double *)calloc(ndof * ndof, sizeof(double));
    double *F = (double *)calloc(ndof, sizeof(double));
    double *U = (double *)malloc(ndof * sizeof(double));
    
    for (int ey = 0; ey < ny; ey++) {
        for (int ex = 0; ex < nx; ex++) {
            int nodes[4] = {
                ey * nnx + ex,
                ey * nnx + ex + 1,
                (ey + 1) * nnx + ex + 1,
                (ey + 1) * nnx + ex
            };
            int dofs[8];
            for (int k = 0; k < 4; k++) {
                dofs[2 * k] = 2 * nodes[k];
                dofs[2 * k + 1] = 2 * nodes[k] + 1;
            }
            for (int r = 0; r < 8; r++) {
                for (int cc = 0; cc < 8; cc++) {
                    K[dofs[r] * ndof + dofs[cc]] += Ke[r][cc];
                }
            }
        }
    }
    
    double fy = total_y / nny;
    for (int i = 0; i < N; i++) {
        if (i % nnx == nx) {
            F[2 * i + 1] += fy;
        }
    }
    
    for (int i = 0; i < N; i++) {
        if (i % nnx == 0) {
            int d1 = 2 * i;
            int d2 = 2 * i + 1;
            for (int j = 0; j < ndof; j++) {
                K[d1 * ndof + j] = 0.0;
                K[j * ndof + d1] = 0.0;
                K[d2 * ndof + j] = 0.0;
                K[j * ndof + d2] = 0.0;
            }
            K[d1 * ndof + d1] = 1.0;
            K[d2 * ndof + d2] = 1.0;
            F[d1] = 0.0;
            F[d2] = 0.0;
        }
    }
    
    solve(K, F, U, ndof);
    
    double sum_uy = 0.0;
    int count = 0;
    for (int i = 0; i < N; i++) {
        if (i % nnx == nx) {
            sum_uy += U[2 * i + 1];
            count++;
        }
    }
    double tip_uy = fabs(sum_uy / count);
    
    printf("tip_uy=%.10g\n", tip_uy);
    
    free(K);
    free(F);
    free(U);
    
    return 0;
}