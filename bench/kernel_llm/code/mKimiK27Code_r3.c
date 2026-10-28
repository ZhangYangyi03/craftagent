#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 6) return 1;
    double L, H, total_y;
    int nx, ny;
    L = atof(argv[1]);
    H = atof(argv[2]);
    nx = atoi(argv[3]);
    ny = atoi(argv[4]);
    total_y = atof(argv[5]);

    const double E = 210000.0;
    const double nu = 0.3;
    const double factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu));
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

    int nnode = (nx + 1) * (ny + 1);
    int nelem = nx * ny;
    double *x = (double *)malloc(nnode * sizeof(double));
    double *y = (double *)malloc(nnode * sizeof(double));
    int *elem = (int *)malloc(nelem * 4 * sizeof(int));

    double dx = L / nx;
    double dy = H / ny;

    for (int j = 0; j <= ny; j++) {
        for (int i = 0; i <= nx; i++) {
            int node = j * (nx + 1) + i;
            x[node] = i * dx;
            y[node] = j * dy;
        }
    }

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int e = j * nx + i;
            elem[4 * e + 0] = j * (nx + 1) + i;
            elem[4 * e + 1] = j * (nx + 1) + i + 1;
            elem[4 * e + 2] = (j + 1) * (nx + 1) + i + 1;
            elem[4 * e + 3] = (j + 1) * (nx + 1) + i;
        }
    }

    int ndof = 2 * nnode;
    double *K = (double *)calloc(ndof * ndof, sizeof(double));
    double *F = (double *)calloc(ndof, sizeof(double));

    double gp[2] = {-1.0 / sqrt(3.0), 1.0 / sqrt(3.0)};
    double gw[2] = {1.0, 1.0};

    for (int e = 0; e < nelem; e++) {
        int n[4];
        for (int i = 0; i < 4; i++) n[i] = elem[4 * e + i];
        double xe[4], ye[4];
        for (int i = 0; i < 4; i++) {
            xe[i] = x[n[i]];
            ye[i] = y[n[i]];
        }

        double ke[8][8] = {0};

        for (int gx = 0; gx < 2; gx++) {
            for (int gy = 0; gy < 2; gy++) {
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

                double J[2][2] = {0};
                for (int i = 0; i < 4; i++) {
                    J[0][0] += dN[i][0] * xe[i];
                    J[0][1] += dN[i][0] * ye[i];
                    J[1][0] += dN[i][1] * xe[i];
                    J[1][1] += dN[i][1] * ye[i];
                }
                double detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0];
                double invJ[2][2];
                invJ[0][0] =  J[1][1] / detJ;
                invJ[0][1] = -J[0][1] / detJ;
                invJ[1][0] = -J[1][0] / detJ;
                invJ[1][1] =  J[0][0] / detJ;

                double dNdX[4][2];
                for (int i = 0; i < 4; i++) {
                    dNdX[i][0] = invJ[0][0] * dN[i][0] + invJ[0][1] * dN[i][1];
                    dNdX[i][1] = invJ[1][0] * dN[i][0] + invJ[1][1] * dN[i][1];
                }

                double B[3][8] = {0};
                for (int i = 0; i < 4; i++) {
                    B[0][2 * i + 0] = dNdX[i][0];
                    B[1][2 * i + 1] = dNdX[i][1];
                    B[2][2 * i + 0] = dNdX[i][1];
                    B[2][2 * i + 1] = dNdX[i][0];
                }

                for (int a = 0; a < 8; a++) {
                    for (int b = 0; b < 8; b++) {
                        double sum = 0.0;
                        for (int k = 0; k < 3; k++) {
                            for (int l = 0; l < 3; l++) {
                                sum += B[k][a] * D[k][l] * B[l][b];
                            }
                        }
                        ke[a][b] += sum * detJ * gw[gx] * gw[gy];
                    }
                }
            }
        }

        for (int a = 0; a < 4; a++) {
            for (int d = 0; d < 2; d++) {
                int row = 2 * n[a] + d;
                for (int b = 0; b < 4; b++) {
                    for (int c = 0; c < 2; c++) {
                        int col = 2 * n[b] + c;
                        K[row * ndof + col] += ke[2 * a + d][2 * b + c];
                    }
                }
            }
        }
    }

    double *fixed = (double *)calloc(ndof, sizeof(double));
    for (int j = 0; j <= ny; j++) {
        int node = j * (nx + 1) + 0;
        fixed[2 * node + 0] = 1.0;
        fixed[2 * node + 1] = 1.0;
    }

    double *right_nodes = (double *)malloc((ny + 1) * sizeof(double));
    int nright = 0;
    for (int j = 0; j <= ny; j++) {
        int node = j * (nx + 1) + nx;
        right_nodes[nright++] = node;
    }
    double node_force = total_y / (double)nright;
    for (int i = 0; i < nright; i++) {
        F[2 * (int)right_nodes[i] + 1] = node_force;
    }
    free(right_nodes);

    double penalty = 1e12;
    for (int i = 0; i < ndof; i++) {
        if (fixed[i] > 0.5) {
            K[i * ndof + i] += penalty;
            F[i] = 0.0;
        }
    }

    double *A = (double *)malloc(ndof * ndof * sizeof(double));
    for (int i = 0; i < ndof * ndof; i++) A[i] = K[i];
    double *b = (double *)malloc(ndof * sizeof(double));
    for (int i = 0; i < ndof; i++) b[i] = F[i];

    int *piv = (int *)malloc(ndof * sizeof(int));
    for (int i = 0; i < ndof; i++) piv[i] = i;

    for (int col = 0; col < ndof; col++) {
        int maxrow = col;
        double maxval = fabs(A[col * ndof + col]);
        for (int row = col + 1; row < ndof; row++) {
            double val = fabs(A[row * ndof + col]);
            if (val > maxval) {
                maxval = val;
                maxrow = row;
            }
        }
        if (maxrow != col) {
            for (int k = 0; k < ndof; k++) {
                double tmp = A[col * ndof + k];
                A[col * ndof + k] = A[maxrow * ndof + k];
                A[maxrow * ndof + k] = tmp;
            }
            double tmp = b[col];
            b[col] = b[maxrow];
            b[maxrow] = tmp;
            int itmp = piv[col];
            piv[col] = piv[maxrow];
            piv[maxrow] = itmp;
        }
        for (int row = col + 1; row < ndof; row++) {
            double factor2 = A[row * ndof + col] / A[col * ndof + col];
            for (int k = col; k < ndof; k++) {
                A[row * ndof + k] -= factor2 * A[col * ndof + k];
            }
            b[row] -= factor2 * b[col];
        }
    }

    double *u = (double *)malloc(ndof * sizeof(double));
    for (int i = ndof - 1; i >= 0; i--) {
        double sum = b[i];
        for (int j = i + 1; j < ndof; j++) {
            sum -= A[i * ndof + j] * u[j];
        }
        u[i] = sum / A[i * ndof + i];
    }

    double avg = 0.0;
    for (int j = 0; j <= ny; j++) {
        int node = j * (nx + 1) + nx;
        avg += u[2 * node + 1];
    }
    avg /= (ny + 1);
    double result = fabs(avg);

    printf("tip_uy=%.10g\n", result);

    free(x);
    free(y);
    free(elem);
    free(K);
    free(F);
    free(fixed);
    free(A);
    free(b);
    free(piv);
    free(u);

    return 0;
}