#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int main(int argc, char *argv[]) {
    double L = atof(argv[1]);
    double H = atof(argv[2]);
    int nx = atoi(argv[3]);
    int ny = atoi(argv[4]);
    double total_y = atof(argv[5]);

    double E = 210000.0;
    double nu = 0.3;

    int nnx = nx + 1;
    int nny = ny + 1;
    int num_nodes = nnx * nny;
    int ndof = 2 * num_nodes;

    double dx = L / nx;
    double dy = H / ny;

    double cf = E / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double D[3][3] = {
        {cf * (1.0 - nu), cf * nu, 0.0},
        {cf * nu, cf * (1.0 - nu), 0.0},
        {0.0, 0.0, cf * (1.0 - 2.0 * nu) / 2.0}
    };

    double *K = (double *)calloc((size_t)ndof * ndof, sizeof(double));
    double *F = (double *)calloc(ndof, sizeof(double));

    double gp = 1.0 / sqrt(3.0);
    double gauss_pts[2] = {-gp, gp};

    for (int ey = 0; ey < ny; ey++) {
        for (int ex = 0; ex < nx; ex++) {
            int n[4];
            n[0] = ey * nnx + ex;
            n[1] = ey * nnx + ex + 1;
            n[2] = (ey + 1) * nnx + ex + 1;
            n[3] = (ey + 1) * nnx + ex;

            double Ke[8][8];
            memset(Ke, 0, sizeof(Ke));

            for (int gi = 0; gi < 2; gi++) {
                for (int gj = 0; gj < 2; gj++) {
                    double xi = gauss_pts[gi];
                    double eta = gauss_pts[gj];

                    double dNdxi[4], dNdeta[4];
                    dNdxi[0] = -(1.0 - eta) / 4.0;
                    dNdxi[1] =  (1.0 - eta) / 4.0;
                    dNdxi[2] =  (1.0 + eta) / 4.0;
                    dNdxi[3] = -(1.0 + eta) / 4.0;

                    dNdeta[0] = -(1.0 - xi) / 4.0;
                    dNdeta[1] = -(1.0 + xi) / 4.0;
                    dNdeta[2] =  (1.0 + xi) / 4.0;
                    dNdeta[3] =  (1.0 - xi) / 4.0;

                    double dNdx[4], dNdy[4];
                    for (int k = 0; k < 4; k++) {
                        dNdx[k] = dNdxi[k] * 2.0 / dx;
                        dNdy[k] = dNdeta[k] * 2.0 / dy;
                    }

                    double B[3][8];
                    memset(B, 0, sizeof(B));
                    for (int k = 0; k < 4; k++) {
                        B[0][2 * k]     = dNdx[k];
                        B[1][2 * k + 1] = dNdy[k];
                        B[2][2 * k]     = dNdy[k];
                        B[2][2 * k + 1] = dNdx[k];
                    }

                    double DB[3][8];
                    memset(DB, 0, sizeof(DB));
                    for (int i = 0; i < 3; i++)
                        for (int j = 0; j < 8; j++)
                            for (int m = 0; m < 3; m++)
                                DB[i][j] += D[i][m] * B[m][j];

                    double detJ = (dx / 2.0) * (dy / 2.0);
                    for (int i = 0; i < 8; i++)
                        for (int j = 0; j < 8; j++)
                            for (int m = 0; m < 3; m++)
                                Ke[i][j] += B[m][i] * DB[m][j] * detJ;
                }
            }

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    int di = 2 * n[i];
                    int dj = 2 * n[j];
                    K[di * ndof + dj]         += Ke[2 * i][2 * j];
                    K[di * ndof + dj + 1]     += Ke[2 * i][2 * j + 1];
                    K[(di + 1) * ndof + dj]     += Ke[2 * i + 1][2 * j];
                    K[(di + 1) * ndof + dj + 1] += Ke[2 * i + 1][2 * j + 1];
                }
            }
        }
    }

    double fy = total_y / (ny + 1);
    for (int j = 0; j <= ny; j++) {
        int node = j * nnx + nx;
        F[2 * node + 1] += fy;
    }

    for (int j = 0; j <= ny; j++) {
        int node = j * nnx;
        int d0 = 2 * node;
        int d1 = 2 * node + 1;
        for (int k = 0; k < ndof; k++) {
            K[d0 * ndof + k] = 0.0;
            K[k * ndof + d0] = 0.0;
            K[d1 * ndof + k] = 0.0;
            K[k * ndof + d1] = 0.0;
        }
        K[d0 * ndof + d0] = 1.0;
        K[d1 * ndof + d1] = 1.0;
        F[d0] = 0.0;
        F[d1] = 0.0;
    }

    double *u = (double *)calloc(ndof, sizeof(double));

    for (int col = 0; col < ndof; col++) {
        int max_row = col;
        double max_val = fabs(K[col * ndof + col]);
        for (int row = col + 1; row < ndof; row++) {
            double val = fabs(K[row * ndof + col]);
            if (val > max_val) {
                max_val = val;
                max_row = row;
            }
        }
        if (max_row != col) {
            for (int k = col; k < ndof; k++) {
                double tmp = K[col * ndof + k];
                K[col * ndof + k] = K[max_row * ndof + k];
                K[max_row * ndof + k] = tmp;
            }
            double tmp = F[col];
            F[col] = F[max_row];
            F[max_row] = tmp;
        }
        double pivot = K[col * ndof + col];
        for (int row = col + 1; row < ndof; row++) {
            double fac = K[row * ndof + col] / pivot;
            for (int k = col + 1; k < ndof; k++)
                K[row * ndof + k] -= fac * K[col * ndof + k];
            F[row] -= fac * F[col];
            K[row * ndof + col] = 0.0;
        }
    }

    for (int i = ndof - 1; i >= 0; i--) {
        double s = F[i];
        for (int j = i + 1; j < ndof; j++)
            s -= K[i * ndof + j] * u[j];
        u[i] = s / K[i * ndof + i];
    }

    double sum_uy = 0.0;
    for (int j = 0; j <= ny; j++) {
        int node = j * nnx + nx;
        sum_uy += u[2 * node + 1];
    }
    double tip_uy = fabs(sum_uy / (ny + 1));

    printf("tip_uy=%.10g\n", tip_uy);

    free(K);
    free(F);
    free(u);
    return 0;
}