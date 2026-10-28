#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double E_ = 210000.0, nu_ = 0.3;

static int node_id(int i, int j, int nx) { return j * (nx + 1) + i; }

int main(int argc, char **argv) {
    if (argc < 6) return 1;
    double L = atof(argv[1]);
    double H = atof(argv[2]);
    int nx = atoi(argv[3]);
    int ny = atoi(argv[4]);
    double total_y = atof(argv[5]);
    if (nx < 1 || ny < 1) return 1;

    int nn = (nx + 1) * (ny + 1);
    int ndof = 2 * nn;
    double *X = malloc(sizeof(double) * nn);
    double *Y = malloc(sizeof(double) * nn);
    for (int j = 0; j <= ny; j++)
        for (int i = 0; i <= nx; i++) {
            int n = node_id(i, j, nx);
            X[n] = L * (double)i / (double)nx;
            Y[n] = H * (double)j / (double)ny;
        }

    double *K = calloc((size_t)ndof * ndof, sizeof(double));
    double *F = calloc(ndof, sizeof(double));
    if (!K || !F) return 1;

    /* Plane strain elasticity matrix */
    double c = E_ / ((1.0 + nu_) * (1.0 - 2.0 * nu_));
    double D[3][3];
    D[0][0] = c * (1.0 - nu_);
    D[0][1] = c * nu_;
    D[1][0] = D[0][1];
    D[1][1] = D[0][0];
    D[2][2] = c * (1.0 - 2.0 * nu_) * 0.5;
    D[0][2] = D[1][2] = D[2][0] = D[2][1] = 0.0;

    double gp = 1.0 / sqrt(3.0);
    double gx[2] = {-gp, gp}, gw[2] = {1.0, 1.0};

    for (int ej = 0; ej < ny; ej++)
        for (int ei = 0; ei < nx; ei++) {
            int en[4];
            en[0] = node_id(ei, ej, nx);
            en[1] = node_id(ei + 1, ej, nx);
            en[2] = node_id(ei + 1, ej + 1, nx);
            en[3] = node_id(ei, ej + 1, nx);
            double ke[8][8] = {{0}};
            for (int a = 0; a < 2; a++)
                for (int b = 0; b < 2; b++) {
                    double xi = gx[a], eta = gx[b];
                    double dNdx[4], dNdy[4];
                    double dN[4][2] = {{-(1.0 - eta), -(1.0 - xi)},
                                       {(1.0 - eta), -(1.0 + xi)},
                                       {(1.0 + eta), (1.0 + xi)},
                                       {-(1.0 + eta), (1.0 + xi)}};
                    double J[2][2] = {{0, 0}, {0, 0}};
                    for (int m = 0; m < 4; m++) {
                        J[0][0] += dN[m][0] * X[en[m]];
                        J[0][1] += dN[m][0] * Y[en[m]];
                        J[1][0] += dN[m][1] * X[en[m]];
                        J[1][1] += dN[m][1] * Y[en[m]];
                    }
                    double detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0];
                    double inv = 1.0 / detJ;
                    double Ji[2][2] = {{J[1][1] * inv, -J[0][1] * inv},
                                       {-J[1][0] * inv, J[0][0] * inv}};
                    for (int m = 0; m < 4; m++) {
                        dNdx[m] = Ji[0][0] * dN[m][0] + Ji[0][1] * dN[m][1];
                        dNdy[m] = Ji[1][0] * dN[m][0] + Ji[1][1] * dN[m][1];
                    }
                    double w = gw[a] * gw[b] * detJ;
                    for (int m = 0; m < 4; m++)
                        for (int n = 0; n < 4; n++) {
                            double B[3][4];
                            B[0][0] = dNdx[m]; B[0][1] = 0.0;      B[0][2] = dNdx[n]; B[0][3] = 0.0;
                            B[1][0] = 0.0;     B[1][1] = dNdy[m]; B[1][2] = 0.0;     B[1][3] = dNdy[n];
                            B[2][0] = dNdy[m]; B[2][1] = dNdx[m]; B[2][2] = dNdy[n]; B[2][3] = dNdx[n];
                            double DB[3][4];
                            for (int r = 0; r < 3; r++)
                                for (int s = 0; s < 4; s++) {
                                    double s2 = 0.0;
                                    for (int t = 0; t < 3; t++) s2 += D[r][t] * B[t][s];
                                    DB[r][s] = s2;
                                }
                            int rm = 2 * m, rn = 2 * n;
                            for (int r = 0; r < 2; r++)
                                for (int s = 0; s < 2; s++) {
                                    double v = 0.0;
                                    for (int t = 0; t < 3; t++) v += B[t][2 * r + 0] * DB[t][2 * s + 0] * 0.0;
                                    (void)v;
                                    double val = 0.0;
                                    for (int t = 0; t < 3; t++)
                                        val += B[t][2 * r] * DB[t][2 * s];
                                    ke[rm + r][rn + s] += val * w;
                                }
                        }
                }
            for (int m = 0; m < 4; m++)
                for (int n = 0; n < 4; n++) {
                    int dm = 2 * en[m], dn = 2 * en[n];
                    for (int r = 0; r < 2; r++)
                        for (int s = 0; s < 2; s++)
                            K[(size_t)(dm + r) * ndof + (dn + s)] += ke[2 * m + r][2 * n + s];
                }
        }

    /* Loads on x = L edge */
    double fy = total_y / (double)(ny + 1);
    for (int j = 0; j <= ny; j++) {
        int n = node_id(nx, j, nx);
        F[2 * n + 1] += fy;
    }

    /* Dirichlet: x = 0 edge fixed */
    for (int j = 0; j <= ny; j++) {
        int n = node_id(0, j, nx);
        int d0 = 2 * n, d1 = 2 * n + 1;
        for (int col = 0; col < ndof; col++) {
            K[(size_t)d0 * ndof + col] = 0.0;
            K[(size_t)d1 * ndof + col] = 0.0;
        }
        for (int row = 0; row < ndof; row++) {
            K[(size_t)row * ndof + d0] = 0.0;
            K[(size_t)row * ndof + d1] = 0.0;
        }
        K[(size_t)d0 * ndof + d0] = 1.0;
        K[(size_t)d1 * ndof + d1] = 1.0;
        F[d0] = 0.0;
        F[d1] = 0.0;
    }

    /* Gaussian elimination with partial pivoting */
    for (int k = 0; k < ndof; k++) {
        int p = k;
        double mx = fabs(K[(size_t)k * ndof + k]);
        for (int r = k + 1; r < ndof; r++) {
            double v = fabs(K[(size_t)r * ndof + k]);
            if (v > mx) { mx = v; p = r; }
        }
        if (mx == 0.0) continue;
        if (p != k) {
            for (int col = 0; col < ndof; col++) {
                double t = K[(size_t)k * ndof + col];
                K[(size_t)k * ndof + col] = K[(size_t)p * ndof + col];
                K[(size_t)p * ndof + col] = t;
            }
            double t = F[k]; F[k] = F[p]; F[p] = t;
        }
        double piv = K[(size_t)k * ndof + k];
        for (int r = k + 1; r < ndof; r++) {
            double f = K[(size_t)r * ndof + k] / piv;
            if (f == 0.0) continue;
            K[(size_t)r * ndof + k] = 0.0;
            for (int col = k + 1; col < ndof; col++)
                K[(size_t)r * ndof + col] -= f * K[(size_t)k * ndof + col];
            F[r] -= f * F[k];
        }
    }
    double *U = calloc(ndof, sizeof(double));
    for (int k = ndof - 1; k >= 0; k--) {
        double s = F[k];
        for (int col = k + 1; col < ndof; col++)
            s -= K[(size_t)k * ndof + col] * U[col];
        double piv = K[(size_t)k * ndof + k];
        U[k] = (piv != 0.0) ? s / piv : 0.0;
    }

    double sum = 0.0;
    for (int j = 0; j <= ny; j++) {
        int n = node_id(nx, j, nx);
        sum += U[2 * n + 1];
    }
    double avg = fabs(sum / (double)(ny + 1));
    printf("tip_uy=%.10g\n", avg);

    free(X); free(Y); free(K); free(F); free(U);
    return 0;
}