#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void solve_linear_system(int n, double *A, double *b) {
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double max_val = fabs(A[col * n + col]);
        for (int row = col + 1; row < n; ++row) {
            double val = fabs(A[row * n + col]);
            if (val > max_val) {
                max_val = val;
                pivot = row;
            }
        }
        if (max_val < 1e-14) {
            continue;
        }
        if (pivot != col) {
            for (int j = 0; j < n; ++j) {
                double tmp = A[col * n + j];
                A[col * n + j] = A[pivot * n + j];
                A[pivot * n + j] = tmp;
            }
            double tmp = b[col];
            b[col] = b[pivot];
            b[pivot] = tmp;
        }
        for (int row = col + 1; row < n; ++row) {
            double factor = A[row * n + col] / A[col * n + col];
            A[row * n + col] = 0.0;
            for (int j = col + 1; j < n; ++j) {
                A[row * n + j] -= factor * A[col * n + j];
            }
            b[row] -= factor * b[col];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= A[i * n + j] * b[j];
        }
        b[i] = sum / A[i * n + i];
    }
}

int main(int argc, char **argv) {
    if (argc != 6) {
        return 1;
    }

    double L = atof(argv[1]);
    double H = atof(argv[2]);
    int nx = atoi(argv[3]);
    int ny = atoi(argv[4]);
    double total_y = atof(argv[5]);

    if (nx < 1 || ny < 1) {
        return 1;
    }

    int nnodes = (nx + 1) * (ny + 1);
    int ndof = 2 * nnodes;

    double *K = calloc((size_t)ndof * ndof, sizeof(double));
    double *F = calloc(ndof, sizeof(double));
    int *fixed = calloc(ndof, sizeof(int));
    if (!K || !F || !fixed) {
        free(K);
        free(F);
        free(fixed);
        return 1;
    }

    double E = 210000.0;
    double nu = 0.3;
    double fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double D[3][3] = {
        { fac * (1.0 - nu), fac * nu, 0.0 },
        { fac * nu, fac * (1.0 - nu), 0.0 },
        { 0.0, 0.0, fac * (0.5 - nu) }
    };

    double gauss_pt = 0.577350269189626;

    for (int ex = 0; ex < nx; ++ex) {
        for (int ey = 0; ey < ny; ++ey) {
            int n[4] = {
                ey * (nx + 1) + ex,
                ey * (nx + 1) + ex + 1,
                (ey + 1) * (nx + 1) + ex + 1,
                (ey + 1) * (nx + 1) + ex
            };

            double x[4], y[4];
            double dx = L / nx;
            double dy = H / ny;
            x[0] = ex * dx;         y[0] = ey * dy;
            x[1] = (ex + 1) * dx;   y[1] = ey * dy;
            x[2] = (ex + 1) * dx;   y[2] = (ey + 1) * dy;
            x[3] = ex * dx;         y[3] = (ey + 1) * dy;

            double Ke[8][8] = {{0.0}};

            for (int gi = 0; gi < 2; ++gi) {
                for (int gj = 0; gj < 2; ++gj) {
                    double xi = (gi == 0) ? -gauss_pt : gauss_pt;
                    double eta = (gj == 0) ? -gauss_pt : gauss_pt;
                    double w = 1.0;

                    double dN_dxi[4], dN_deta[4];
                    dN_dxi[0] = -(1.0 - eta) * 0.25;
                    dN_dxi[1] =  (1.0 - eta) * 0.25;
                    dN_dxi[2] =  (1.0 + eta) * 0.25;
                    dN_dxi[3] = -(1.0 + eta) * 0.25;

                    dN_deta[0] = -(1.0 - xi) * 0.25;
                    dN_deta[1] = -(1.0 + xi) * 0.25;
                    dN_deta[2] =  (1.0 + xi) * 0.25;
                    dN_deta[3] =  (1.0 - xi) * 0.25;

                    double J[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
                    for (int i = 0; i < 4; ++i) {
                        J[0][0] += dN_dxi[i] * x[i];
                        J[0][1] += dN_dxi[i] * y[i];
                        J[1][0] += dN_deta[i] * x[i];
                        J[1][1] += dN_deta[i] * y[i];
                    }

                    double detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0];
                    if (detJ <= 0.0) {
                        continue;
                    }

                    double invJ[2][2] = {
                        {  J[1][1] / detJ, -J[0][1] / detJ },
                        { -J[1][0] / detJ,  J[0][0] / detJ }
                    };

                    double dN_dx[4], dN_dy[4];
                    for (int i = 0; i < 4; ++i) {
                        dN_dx[i] = invJ[0][0] * dN_dxi[i] + invJ[0][1] * dN_deta[i];
                        dN_dy[i] = invJ[1][0] * dN_dxi[i] + invJ[1][1] * dN_deta[i];
                    }

                    double B[3][8] = {{0.0}};
                    for (int i = 0; i < 4; ++i) {
                        B[0][2 * i] = dN_dx[i];
                        B[1][2 * i + 1] = dN_dy[i];
                        B[2][2 * i] = dN_dy[i];
                        B[2][2 * i + 1] = dN_dx[i];
                    }

                    for (int i = 0; i < 8; ++i) {
                        for (int j = 0; j < 8; ++j) {
                            double sum = 0.0;
                            for (int k = 0; k < 3; ++k) {
                                double temp = 0.0;
                                for (int l = 0; l < 3; ++l) {
                                    temp += D[k][l] * B[l][j];
                                }
                                sum += B[k][i] * temp;
                            }
                            Ke[i][j] += sum * detJ * w;
                        }
                    }
                }
            }

            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) {
                    int row0 = 2 * n[a];
                    int col0 = 2 * n[b];
                    K[(row0) * ndof + col0] += Ke[2 * a][2 * b];
                    K[(row0) * ndof + col0 + 1] += Ke[2 * a][2 * b + 1];
                    K[(row0 + 1) * ndof + col0] += Ke[2 * a + 1][2 * b];
                    K[(row0 + 1) * ndof + col0 + 1] += Ke[2 * a + 1][2 * b + 1];
                }
            }
        }
    }

    double per_node = total_y / (ny + 1);
    for (int iy = 0; iy <= ny; ++iy) {
        int id = iy * (nx + 1) + nx;
        F[2 * id + 1] += per_node;
    }

    for (int iy = 0; iy <= ny; ++iy) {
        int id = iy * (nx + 1);
        fixed[2 * id] = 1;
        fixed[2 * id + 1] = 1;
    }

    for (int i = 0; i < ndof; ++i) {
        if (fixed[i]) {
            for (int j = 0; j < ndof; ++j) {
                K[i * ndof + j] = 0.0;
                K[j * ndof + i] = 0.0;
            }
            K[i * ndof + i] = 1.0;
            F[i] = 0.0;
        }
    }

    solve_linear_system(ndof, K, F);

    double sum_uy = 0.0;
    for (int iy = 0; iy <= ny; ++iy) {
        int id = iy * (nx + 1) + nx;
        sum_uy += F[2 * id + 1];
    }
    double avg_abs = fabs(sum_uy / (ny + 1));

    printf("tip_uy=%.10g\n", avg_abs);

    free(K);
    free(F);
    free(fixed);
    return 0;
}