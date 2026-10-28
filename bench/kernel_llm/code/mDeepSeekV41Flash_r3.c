#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv){
    if(argc != 6) return 1;
    double L = atof(argv[1]);
    double H = atof(argv[2]);
    int nx = atoi(argv[3]);
    int ny = atoi(argv[4]);
    double total_y = atof(argv[5]);

    if(nx <= 0 || ny <= 0) return 1;

    int nnx = nx + 1;
    int nny = ny + 1;
    int nnodes = nnx * nny;
    int ndof = 2 * nnodes;

    double *X = (double*)malloc((size_t)nnodes * sizeof(double));
    double *Y = (double*)malloc((size_t)nnodes * sizeof(double));
    if(!X || !Y) return 1;
    for(int j = 0; j < nny; j++){
        for(int i = 0; i < nnx; i++){
            int n = j * nnx + i;
            X[n] = L * (double)i / (double)nx;
            Y[n] = H * (double)j / (double)ny;
        }
    }

    /* Plane strain isotropic elasticity */
    double E = 210000.0;
    double nu = 0.3;
    double ccoef = E / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double D[3][3];
    D[0][0] = ccoef * (1.0 - nu); D[0][1] = ccoef * nu;           D[0][2] = 0.0;
    D[1][0] = ccoef * nu;         D[1][1] = ccoef * (1.0 - nu);   D[1][2] = 0.0;
    D[2][0] = 0.0;                D[2][1] = 0.0;                  D[2][2] = ccoef * (1.0 - 2.0 * nu) / 2.0;

    double *K = (double*)calloc((size_t)ndof * (size_t)ndof, sizeof(double));
    if(!K) return 1;

    double gp[2] = { -1.0 / sqrt(3.0), 1.0 / sqrt(3.0) };
    double gw[2] = { 1.0, 1.0 };

    double xi_n[4]  = { -1.0,  1.0,  1.0, -1.0 };
    double eta_n[4] = { -1.0, -1.0,  1.0,  1.0 };

    for(int ej = 0; ej < ny; ej++){
        for(int ei = 0; ei < nx; ei++){
            int nid[4];
            nid[0] = ej       * nnx + ei;
            nid[1] = ej       * nnx + ei + 1;
            nid[2] = (ej + 1) * nnx + ei + 1;
            nid[3] = (ej + 1) * nnx + ei;

            double ex[4], ey[4];
            for(int k = 0; k < 4; k++){ ex[k] = X[nid[k]]; ey[k] = Y[nid[k]]; }

            double Ke[8][8];
            memset(Ke, 0, sizeof(Ke));

            for(int gxi = 0; gxi < 2; gxi++){
                for(int geta = 0; geta < 2; geta++){
                    double xi  = gp[gxi];
                    double eta = gp[geta];
                    double dN_dxi[4], dN_deta[4];
                    for(int k = 0; k < 4; k++){
                        dN_dxi[k]  = 0.25 * xi_n[k]  * (1.0 + eta_n[k] * eta);
                        dN_deta[k] = 0.25 * eta_n[k] * (1.0 + xi_n[k]  * xi);
                    }

                    double J11 = 0.0, J12 = 0.0, J21 = 0.0, J22 = 0.0;
                    for(int k = 0; k < 4; k++){
                        J11 += dN_dxi[k]  * ex[k];
                        J12 += dN_dxi[k]  * ey[k];
                        J21 += dN_deta[k] * ex[k];
                        J22 += dN_deta[k] * ey[k];
                    }
                    double detJ = J11 * J22 - J12 * J21;

                    double dN_dx[4], dN_dy[4];
                    for(int k = 0; k < 4; k++){
                        dN_dx[k] = ( J22 * dN_dxi[k] - J12 * dN_deta[k]) / detJ;
                        dN_dy[k] = (-J21 * dN_dxi[k] + J11 * dN_deta[k]) / detJ;
                    }

                    double B[3][8];
                    memset(B, 0, sizeof(B));
                    for(int k = 0; k < 4; k++){
                        B[0][2*k]     = dN_dx[k];
                        B[1][2*k + 1] = dN_dy[k];
                        B[2][2*k]     = dN_dy[k];
                        B[2][2*k + 1] = dN_dx[k];
                    }

                    double w = gw[gxi] * gw[geta] * detJ;
                    for(int a = 0; a < 8; a++){
                        for(int b = 0; b < 8; b++){
                            double s = 0.0;
                            for(int p = 0; p < 3; p++)
                                for(int q = 0; q < 3; q++)
                                    s += B[p][a] * D[p][q] * B[q][b];
                            Ke[a][b] += s * w;
                        }
                    }
                }
            }

            for(int a = 0; a < 8; a++){
                int Ia = nid[a / 2] * 2 + (a % 2);
                for(int b = 0; b < 8; b++){
                    int Ib = nid[b / 2] * 2 + (b % 2);
                    K[(size_t)Ia * ndof + Ib] += Ke[a][b];
                }
            }
        }
    }

    double *F = (double*)calloc((size_t)ndof, sizeof(double));
    if(!F) return 1;

    double perNode = total_y / (double)nny;
    for(int j = 0; j < nny; j++){
        int n = j * nnx + (nnx - 1);
        F[n * 2 + 1] += perNode;
    }

    /* Dirichlet BCs: x = 0 edge fully fixed */
    for(int j = 0; j < nny; j++){
        int n = j * nnx + 0;
        for(int d = 0; d < 2; d++){
            int dd = n * 2 + d;
            for(int cc = 0; cc < ndof; cc++) K[(size_t)dd * ndof + cc] = 0.0;
            for(int rr = 0; rr < ndof; rr++) K[(size_t)rr * ndof + dd] = 0.0;
            K[(size_t)dd * ndof + dd] = 1.0;
            F[dd] = 0.0;
        }
    }

    /* Dense Gaussian elimination with partial pivoting */
    int n = ndof;
    for(int col = 0; col < n; col++){
        int piv = col;
        double maxv = fabs(K[(size_t)col * n + col]);
        for(int r = col + 1; r < n; r++){
            double v = fabs(K[(size_t)r * n + col]);
            if(v > maxv){ maxv = v; piv = r; }
        }
        if(piv != col){
            for(int cc = 0; cc < n; cc++){
                double t = K[(size_t)col * n + cc];
                K[(size_t)col * n + cc] = K[(size_t)piv * n + cc];
                K[(size_t)piv * n + cc] = t;
            }
            double t = F[col]; F[col] = F[piv]; F[piv] = t;
        }
        double pv = K[(size_t)col * n + col];
        for(int r = col + 1; r < n; r++){
            double fac = K[(size_t)r * n + col] / pv;
            if(fac == 0.0) continue;
            K[(size_t)r * n + col] = 0.0;
            for(int cc = col + 1; cc < n; cc++)
                K[(size_t)r * n + cc] -= fac * K[(size_t)col * n + cc];
            F[r] -= fac * F[col];
        }
    }
    for(int r = n - 1; r >= 0; r--){
        double s = F[r];
        for(int cc = r + 1; cc < n; cc++) s -= K[(size_t)r * n + cc] * F[cc];
        F[r] = s / K[(size_t)r * n + r];
    }

    double sum = 0.0;
    for(int j = 0; j < nny; j++){
        int nn = j * nnx + (nnx - 1);
        sum += F[nn * 2 + 1];
    }
    double avg = sum / (double)nny;

    printf("tip_uy=%.10g\n", fabs(avg));

    free(X);
    free(Y);
    free(K);
    free(F);
    return 0;
}