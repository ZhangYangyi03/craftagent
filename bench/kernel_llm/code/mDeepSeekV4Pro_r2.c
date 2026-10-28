#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void compute_D(double E, double nu, double D[3][3]) {
    double fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu));
    D[0][0] = D[1][1] = fac * (1.0 - nu);
    D[0][1] = D[1][0] = fac * nu;
    D[0][2] = D[1][2] = D[2][0] = D[2][1] = 0.0;
    D[2][2] = fac * (1.0 - 2.0 * nu) / 2.0;
}

static void compute_B(double dN_dx[4], double dN_dy[4], double B[3][8]) {
    for (int a = 0; a < 4; a++) {
        int c = 2 * a;
        B[0][c] = dN_dx[a];
        B[0][c + 1] = 0.0;
        B[1][c] = 0.0;
        B[1][c + 1] = dN_dy[a];
        B[2][c] = dN_dy[a];
        B[2][c + 1] = dN_dx[a];
    }
}

static void solve_dense(int n, double *A, double *b) {
    for (int k = 0; k < n; k++) {
        int max_row = k;
        double max_val = fabs(A[k * n + k]);
        for (int i = k + 1; i < n; i++) {
            double val = fabs(A[i * n + k]);
            if (val > max_val) {
                max_val = val;
                max_row = i;
            }
        }
        if (max_row != k) {
            for (int j = k; j < n; j++) {
                double tmp = A[k * n + j];
                A[k * n + j] = A[max_row * n + j];
                A[max_row * n + j] = tmp;
            }
            double tmp = b[k];
            b[k] = b[max_row];
            b[max_row] = tmp;
        }
        double pivot = A[k * n + k];
        for (int i = k + 1; i < n; i++) {
            double factor = A[i * n + k] / pivot;
            if (factor != 0.0) {
                A[i * n + k] = 0.0;
                for (int j = k + 1; j < n; j++) {
                    A[i * n + j] -= factor * A[k * n + j];
                }
                b[i] -= factor * b[k];
            }
        }
    }
    for (int i = n - 1; i >= 0; i--) {
        double sum = b[i];
        for (int j = i + 1; j < n; j++) {
            sum -= A[i * n + j] * b[j];
        }
        b[i] = sum / A[i * n + i];
    }
}

int main(int argc, char **argv) {
    if (argc != 6) return 1;
    double L = atof(argv[1]);
    double H = atof(argv[2]);
    int nx = atoi(argv[3]);
    int ny = atoi(argv[4]);
    double total_y = atof(argv[5]);

    int npx = nx + 1;
    int npy = ny + 1;
    int n_nodes = npx * npy;
    int ndof = 2 * n_nodes;

    double dx = L / nx;
    double dy = H / ny;

    double *K = (double *)calloc((size_t)ndof * ndof, sizeof(double));
    double *F = (double *)calloc(ndof, sizeof(double));
    if (!K || !F) {
        free(K);
        free(F);
        return 1;
    }

    double D[3][3];
    compute_D(210000.0, 0.3, D);

    double gauss_pts[2] = { -1.0 / sqrt(3.0), 1.0 / sqrt(3.0) };
    double weights[2] = { 1.0, 1.0 };

    for (int ey = 0; ey < ny; ey++) {
        for (int ex = 0; ex < nx; ex++) {
            int nodes[4];
            nodes[0] = ey * npx + ex;
            nodes[1] = ey * npx + ex + 1;
            nodes[2] = (ey + 1) * npx + ex + 1;
            nodes[3] = (ey + 1) * npx + ex;

            double x[4], y[4];
            for (int a = 0; a < 4; a++) {
                int i = nodes[a] % npx;
                int j = nodes[a] / npx;
                x[a] = i * dx;
                y[a] = j * dy;
            }

            double Ke[64] = {0};

            for (int gi = 0; gi < 2; gi++) {
                for (int gj = 0; gj < 2; gj++) {
                    double xi = gauss_pts[gi];
                    double eta = gauss_pts[gj];
                    double w = weights[gi] * weights[gj];

                    double dN_dxi[4], dN_deta[4];
                    dN_dxi[0] = -0.25 * (1.0 - eta);
                    dN_dxi[1] =  0.25 * (1.0 - eta);
                    dN_dxi[2] =  0.25 * (1.0 + eta);
                    dN_dxi[3] = -0.25 * (1.0 + eta);

                    dN_deta[0] = -0.25 * (1.0 - xi);
                    dN_deta[1] = -0.25 * (1.0 + xi);
                    dN_deta[2] =  0.25 * (1.0 + xi);
                    dN_deta[3] =  0.25 * (1.0 - xi);

                    double J[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
                    for (int a = 0; a < 4; a++) {
                        J[0][0] += dN_dxi[a] * x[a];
                        J[0][1] += dN_deta[a] * x[a];
                        J[1][0] += dN_dxi[a] * y[a];
                        J[1][1] += dN_deta[a] * y[a];
                    }
                    double detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0];

                    double dN_dx[4], dN_dy[4];
                    for (int a = 0; a < 4; a++) {
                        dN_dx[a] = (J[1][1] * dN_dxi[a] - J[0][1] * dN_deta[a]) / detJ;
                        dN_dy[a] = (-J[1][0] * dN_dxi[a] + J[0][0] * dN_deta[a]) / detJ;
                    }

                    double B[3][8] = {{0.0}};
                    compute_B(dN_dx, dN_dy, B);

                    for (int r = 0; r < 8; r++) {
                        for (int s = 0; s < 8; s++) {
                            double val = 0.0;
                            for (int p = 0; p < 3; p++) {
                                for (int q = 0; q < 3; q++) {
                                    val += B[p][r] * D[p][q] * B[q][s];
                                }
                            }
                            Ke[r * 8 + s] += val * detJ * w;
                        }
                    }
                }
            }

            for (int a = 0; a < 8; a++) {
                int na = a / 2;
                int da = a % 2;
                int ga = 2 * nodes[na] + da;
                for (int b = 0; b < 8; b++) {
                    int nb = b / 2;
                    int db = b % 2;
                    int gb = 2 * nodes[nb] + db;
                    K[ga * ndof + gb] += Ke[a * 8 + b];
                }
            }
        }
    }

    double force_per_node = total_y / (ny + 1);
    for (int j = 0; j <= ny; j++) {
        int nid = j * npx + nx;
        int dof = 2 * nid + 1;
        F[dof] += force_per_node;
    }

    for (int j = 0; j <= ny; j++) {
        int nid = j * npx;
        int dof_ux = 2 * nid;
        int dof_uy = 2 * nid + 1;

        for (int r = 0; r < ndof; r++) {
            K[r * ndof + dof_ux] = 0.0;
            K[dof_ux * ndof + r] = 0.0;
            K[r * ndof + dof_uy] = 0.0;
            K[dof_uy * ndof + r] = 0.0;
        }
        K[dof_ux * ndof + dof_ux] = 1.0;
        K[dof_uy * ndof + dof_uy] = 1.0;
        F[dof_ux] = 0.0;
        F[dof_uy] = 0.0;
    }

    solve_dense(ndof, K, F);

    double sum_uy = 0.0;
    for (int j = 0; j <= ny; j++) {
        int nid = j * npx + nx;
        int dof_uy = 2 * nid + 1;
        sum_uy += F[dof_uy];
    }
    double avg_uy = sum_uy / (ny + 1);
    double result = fabs(avg_uy);

    printf("tip_uy=%.10g\n", result);

    free(K);
    free(F);
    return 0;
}