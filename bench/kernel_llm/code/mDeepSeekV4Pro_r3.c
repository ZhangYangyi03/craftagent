#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void solve_linear(int n, double **A, double *b, double *x) {
    for (int k = 0; k < n; k++) {
        int max_row = k;
        double max_val = fabs(A[k][k]);
        for (int i = k + 1; i < n; i++) {
            if (fabs(A[i][k]) > max_val) {
                max_val = fabs(A[i][k]);
                max_row = i;
            }
        }
        if (max_row != k) {
            for (int j = k; j < n; j++) {
                double tmp = A[k][j];
                A[k][j] = A[max_row][j];
                A[max_row][j] = tmp;
            }
            double tmp = b[k];
            b[k] = b[max_row];
            b[max_row] = tmp;
        }
        for (int i = k + 1; i < n; i++) {
            double factor = A[i][k] / A[k][k];
            A[i][k] = 0.0;
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }
    for (int i = n - 1; i >= 0; i--) {
        double sum = b[i];
        for (int j = i + 1; j < n; j++) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        return 1;
    }
    double L = atof(argv[1]);
    double H = atof(argv[2]);
    int nx = atoi(argv[3]);
    int ny = atoi(argv[4]);
    double total_y = atof(argv[5]);

    double E = 210000.0;
    double nu = 0.3;

    double D[3][3] = {{0}};
    double fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu));
    D[0][0] = D[1][1] = fac * (1.0 - nu);
    D[0][1] = D[1][0] = fac * nu;
    D[2][2] = E / (2.0 * (1.0 + nu));

    int nnodes = (nx + 1) * (ny + 1);
    int ndof = 2 * nnodes;

    double *xcoord = (double *)malloc(nnodes * sizeof(double));
    double *ycoord = (double *)malloc(nnodes * sizeof(double));
    for (int j = 0; j <= ny; j++) {
        for (int i = 0; i <= nx; i++) {
            int node = j * (nx + 1) + i;
            xcoord[node] = i * (L / nx);
            ycoord[node] = j * (H / ny);
        }
    }

    double **K = (double **)malloc(ndof * sizeof(double *));
    for (int i = 0; i < ndof; i++) {
        K[i] = (double *)calloc(ndof, sizeof(double));
    }
    double *F = (double *)calloc(ndof, sizeof(double));

    double gp[2] = {-1.0 / sqrt(3.0), 1.0 / sqrt(3.0)};

    for (int ey = 0; ey < ny; ey++) {
        for (int ex = 0; ex < nx; ex++) {
            int n[4];
            n[0] = ey * (nx + 1) + ex;
            n[1] = ey * (nx + 1) + ex + 1;
            n[2] = (ey + 1) * (nx + 1) + ex + 1;
            n[3] = (ey + 1) * (nx + 1) + ex;

            double xe[4], ye[4];
            for (int i = 0; i < 4; i++) {
                xe[i] = xcoord[n[i]];
                ye[i] = ycoord[n[i]];
            }

            double ke[8][8] = {{0}};

            for (int ig = 0; ig < 2; ig++) {
                double xi = gp[ig];
                for (int jg = 0; jg < 2; jg++) {
                    double eta = gp[jg];

                    double dN_dxi[4], dN_deta[4];
                    dN_dxi[0] = -(1.0 - eta) / 4.0;
                    dN_dxi[1] =  (1.0 - eta) / 4.0;
                    dN_dxi[2] =  (1.0 + eta) / 4.0;
                    dN_dxi[3] = -(1.0 + eta) / 4.0;

                    dN_deta[0] = -(1.0 - xi) / 4.0;
                    dN_deta[1] = -(1.0 + xi) / 4.0;
                    dN_deta[2] =  (1.0 + xi) / 4.0;
                    dN_deta[3] =  (1.0 - xi) / 4.0;

                    double dx_dxi = 0.0, dx_deta = 0.0;
                    double dy_dxi = 0.0, dy_deta = 0.0;
                    for (int i = 0; i < 4; i++) {
                        dx_dxi  += dN_dxi[i] * xe[i];
                        dx_deta += dN_deta[i] * xe[i];
                        dy_dxi  += dN_dxi[i] * ye[i];
                        dy_deta += dN_deta[i] * ye[i];
                    }
                    double detJ = dx_dxi * dy_deta - dx_deta * dy_dxi;

                    double dN_dx[4], dN_dy[4];
                    for (int i = 0; i < 4; i++) {
                        dN_dx[i] = (dy_deta * dN_dxi[i] - dy_dxi * dN_deta[i]) / detJ;
                        dN_dy[i] = (-dx_deta * dN_dxi[i] + dx_dxi * dN_deta[i]) / detJ;
                    }

                    double B[3][8] = {{0}};
                    for (int a = 0; a < 4; a++) {
                        int col = 2 * a;
                        B[0][col] = dN_dx[a];
                        B[1][col + 1] = dN_dy[a];
                        B[2][col] = dN_dy[a];
                        B[2][col + 1] = dN_dx[a];
                    }

                    for (int a = 0; a < 4; a++) {
                        for (int b = 0; b < 4; b++) {
                            for (int i = 0; i < 2; i++) {
                                for (int j = 0; j < 2; j++) {
                                    double sum = 0.0;
                                    for (int p = 0; p < 3; p++) {
                                        for (int q = 0; q < 3; q++) {
                                            sum += B[p][2*a+i] * D[p][q] * B[q][2*b+j];
                                        }
                                    }
                                    ke[2*a+i][2*b+j] += sum * detJ;
                                }
                            }
                        }
                    }
                }
            }

            for (int a = 0; a < 4; a++) {
                int gnode_a = n[a];
                for (int i = 0; i < 2; i++) {
                    int row = 2 * gnode_a + i;
                    for (int b = 0; b < 4; b++) {
                        int gnode_b = n[b];
                        for (int j = 0; j < 2; j++) {
                            int col = 2 * gnode_b + j;
                            K[row][col] += ke[2*a+i][2*b+j];
                        }
                    }
                }
            }
        }
    }

    double f_per_node = total_y / (ny + 1);
    for (int j = 0; j <= ny; j++) {
        int node = j * (nx + 1) + nx;
        int dof = 2 * node + 1;
        F[dof] -= f_per_node;
    }

    for (int j = 0; j <= ny; j++) {
        int node = j * (nx + 1);
        for (int dir = 0; dir < 2; dir++) {
            int dof = 2 * node + dir;
            for (int k = 0; k < ndof; k++) {
                K[dof][k] = 0.0;
                K[k][dof] = 0.0;
            }
            K[dof][dof] = 1.0;
            F[dof] = 0.0;
        }
    }

    double *x = (double *)malloc(ndof * sizeof(double));
    solve_linear(ndof, K, F, x);

    double sum_uy = 0.0;
    for (int j = 0; j <= ny; j++) {
        int node = j * (nx + 1) + nx;
        sum_uy += x[2 * node + 1];
    }
    double avg_uy = sum_uy / (ny + 1);

    printf("tip_uy=%.10g\n", fabs(avg_uy));

    free(x);
    for (int i = 0; i < ndof; i++) free(K[i]);
    free(K);
    free(F);
    free(xcoord);
    free(ycoord);

    return 0;
}