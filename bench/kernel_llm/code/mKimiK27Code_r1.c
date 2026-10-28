#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double det2(double a, double b, double c, double d) {
    return a * d - b * c;
}

int main(int argc, char **argv) {
    if (argc < 6) return 1;
    double L = atof(argv[1]);
    double H = atof(argv[2]);
    int nx = atoi(argv[3]);
    int ny = atoi(argv[4]);
    double total_y = atof(argv[5]);

    int nx_nodes = nx + 1;
    int ny_nodes = ny + 1;
    int nnodes = nx_nodes * ny_nodes;
    int nelem = nx * ny;

    double E = 210000.0;
    double nu = 0.3;
    double factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double D[3][3];
    D[0][0] = factor * (1.0 - nu);
    D[0][1] = factor * nu;
    D[0][2] = 0.0;
    D[1][0] = factor * nu;
    D[1][1] = factor * (1.0 - nu);
    D[1][2] = 0.0;
    D[2][0] = 0.0;
    D[2][1] = 0.0;
    D[2][2] = factor * (1.0 - 2.0 * nu) / 2.0;

    double *x = (double *)malloc(nnodes * sizeof(double));
    double *y = (double *)malloc(nnodes * sizeof(double));
    for (int j = 0; j < ny_nodes; ++j) {
        for (int i = 0; i < nx_nodes; ++i) {
            int node = j * nx_nodes + i;
            x[node] = L * i / nx;
            y[node] = H * j / ny;
        }
    }

    double *Ke = (double *)malloc(8 * 8 * sizeof(double));
    double *K = (double *)malloc(2 * nnodes * 2 * nnodes * sizeof(double));
    for (int i = 0; i < 4 * nnodes * nnodes; ++i) K[i] = 0.0;

    double gp[2] = {-1.0 / sqrt(3.0), 1.0 / sqrt(3.0)};
    double gw[2] = {1.0, 1.0};

    for (int ey = 0; ey < ny; ++ey) {
        for (int ex = 0; ex < nx; ++ex) {
            int n1 = ey * nx_nodes + ex;
            int n2 = n1 + 1;
            int n4 = n1 + nx_nodes;
            int n3 = n4 + 1;
            int elem_nodes[4] = {n1, n2, n3, n4};

            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    Ke[i * 8 + j] = 0.0;
                }
            }

            for (int gx = 0; gx < 2; ++gx) {
                for (int gy = 0; gy < 2; ++gy) {
                    double xi = gp[gx];
                    double eta = gp[gy];
                    double dN[4][2];
                    dN[0][0] = -0.25 * (1.0 - eta);
                    dN[0][1] = -0.25 * (1.0 - xi);
                    dN[1][0] =  0.25 * (1.0 - eta);
                    dN[1][1] = -0.25 * (1.0 + xi);
                    dN[2][0] =  0.25 * (1.0 + eta);
                    dN[2][1] =  0.25 * (1.0 + xi);
                    dN[3][0] = -0.25 * (1.0 + eta);
                    dN[3][1] =  0.25 * (1.0 - xi);

                    double J[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
                    for (int a = 0; a < 4; ++a) {
                        J[0][0] += dN[a][0] * x[elem_nodes[a]];
                        J[0][1] += dN[a][0] * y[elem_nodes[a]];
                        J[1][0] += dN[a][1] * x[elem_nodes[a]];
                        J[1][1] += dN[a][1] * y[elem_nodes[a]];
                    }
                    double detJ = det2(J[0][0], J[0][1], J[1][0], J[1][1]);
                    double invJ[2][2];
                    invJ[0][0] =  J[1][1] / detJ;
                    invJ[0][1] = -J[0][1] / detJ;
                    invJ[1][0] = -J[1][0] / detJ;
                    invJ[1][1] =  J[0][0] / detJ;

                    double dNdX[4][2];
                    for (int a = 0; a < 4; ++a) {
                        dNdX[a][0] = invJ[0][0] * dN[a][0] + invJ[0][1] * dN[a][1];
                        dNdX[a][1] = invJ[1][0] * dN[a][0] + invJ[1][1] * dN[a][1];
                    }

                    double B[3][8];
                    for (int a = 0; a < 4; ++a) {
                        B[0][2*a]   = dNdX[a][0];
                        B[0][2*a+1] = 0.0;
                        B[1][2*a]   = 0.0;
                        B[1][2*a+1] = dNdX[a][1];
                        B[2][2*a]   = dNdX[a][1];
                        B[2][2*a+1] = dNdX[a][0];
                    }

                    double DB[3][8];
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 8; ++j) {
                            DB[i][j] = 0.0;
                            for (int k = 0; k < 3; ++k) {
                                DB[i][j] += D[i][k] * B[k][j];
                            }
                        }
                    }

                    double w = detJ * gw[gx] * gw[gy];
                    for (int i = 0; i < 8; ++i) {
                        for (int j = 0; j < 8; ++j) {
                            double val = 0.0;
                            for (int k = 0; k < 3; ++k) {
                                val += B[k][i] * DB[k][j];
                            }
                            Ke[i * 8 + j] += val * w;
                        }
                    }
                }
            }

            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) {
                    for (int d = 0; d < 2; ++d) {
                        for (int e = 0; e < 2; ++e) {
                            int row = 2 * elem_nodes[a] + d;
                            int col = 2 * elem_nodes[b] + e;
                            K[row * (2 * nnodes) + col] += Ke[(2*a+d) * 8 + (2*b+e)];
                        }
                    }
                }
            }
        }
    }

    free(Ke);

    double *F = (double *)malloc(2 * nnodes * sizeof(double));
    for (int i = 0; i < 2 * nnodes; ++i) F[i] = 0.0;

    double node_force = total_y / ny_nodes;
    for (int j = 0; j < ny_nodes; ++j) {
        int node = j * nx_nodes + nx;
        F[2 * node + 1] = node_force;
    }

    int *fixed = (int *)malloc(2 * nnodes * sizeof(int));
    for (int i = 0; i < 2 * nnodes; ++i) fixed[i] = 0;
    for (int j = 0; j < ny_nodes; ++j) {
        int node = j * nx_nodes;
        fixed[2 * node] = 1;
        fixed[2 * node + 1] = 1;
    }

    int ndof = 2 * nnodes;
    for (int i = 0; i < ndof; ++i) {
        if (fixed[i]) {
            for (int j = 0; j < ndof; ++j) {
                K[i * ndof + j] = 0.0;
            }
            K[i * ndof + i] = 1.0;
            F[i] = 0.0;
        }
    }

    double *U = (double *)malloc(ndof * sizeof(double));
    int *piv = (int *)malloc(ndof * sizeof(int));
    for (int i = 0; i < ndof; ++i) piv[i] = i;

    for (int k = 0; k < ndof; ++k) {
        int maxrow = k;
        double maxval = fabs(K[piv[k] * ndof + k]);
        for (int i = k + 1; i < ndof; ++i) {
            double val = fabs(K[piv[i] * ndof + k]);
            if (val > maxval) {
                maxval = val;
                maxrow = i;
            }
        }
        int tmp = piv[k];
        piv[k] = piv[maxrow];
        piv[maxrow] = tmp;

        int pk = piv[k];
        for (int i = k + 1; i < ndof; ++i) {
            int pi = piv[i];
            double factor_lu = K[pi * ndof + k] / K[pk * ndof + k];
            K[pi * ndof + k] = factor_lu;
            for (int j = k + 1; j < ndof; ++j) {
                K[pi * ndof + j] -= factor_lu * K[pk * ndof + j];
            }
        }
    }

    for (int i = 0; i < ndof; ++i) {
        U[i] = F[piv[i]];
    }

    for (int i = 0; i < ndof; ++i) {
        for (int j = 0; j < i; ++j) {
            U[i] -= K[piv[i] * ndof + j] * U[j];
        }
    }

    for (int i = ndof - 1; i >= 0; --i) {
        for (int j = i + 1; j < ndof; ++j) {
            U[i] -= K[piv[i] * ndof + j] * U[j];
        }
        U[i] /= K[piv[i] * ndof + i];
    }

    double sum = 0.0;
    for (int j = 0; j < ny_nodes; ++j) {
        int node = j * nx_nodes + nx;
        sum += U[2 * node + 1];
    }
    double avg = sum / ny_nodes;
    double tip_uy = fabs(avg);

    printf("tip_uy=%.10g\n", tip_uy);

    free(x);
    free(y);
    free(K);
    free(F);
    free(fixed);
    free(U);
    free(piv);

    return 0;
}