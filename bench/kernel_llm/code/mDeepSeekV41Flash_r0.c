#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[]) {
    if (argc < 6) return 1;

    double L = atof(argv[1]);
    double H = atof(argv[2]);
    int nx = atoi(argv[3]);
    int ny = atoi(argv[4]);
    double total_y = atof(argv[5]);

    if (nx < 1 || ny < 1 || L <= 0.0 || H <= 0.0) return 1;

    const double E = 210000.0;
    const double nu = 0.3;

    int nnode = (nx + 1) * (ny + 1);
    int n_dof = 2 * nnode;

    double *K = calloc((size_t)n_dof * (size_t)n_dof, sizeof(double));
    double *F = calloc((size_t)n_dof, sizeof(double));
    double *u = malloc((size_t)n_dof * sizeof(double));
    if (!K || !F || !u) {
        free(K);
        free(F);
        free(u);
        return 1;
    }

    double hx = L / (double)nx;
    double hy = H / (double)ny;

    double c = E / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double D[3][3] = {
        { c * (1.0 - nu), c * nu, 0.0 },
        { c * nu, c * (1.0 - nu), 0.0 },
        { 0.0, 0.0, c * (1.0 - 2.0 * nu) / 2.0 }
    };

    const double gp[2] = { -0.57735026918962576451, 0.57735026918962576451 };
    const double gw[2] = { 1.0, 1.0 };

    for (int ey = 0; ey < ny; ey++) {
        for (int ex = 0; ex < nx; ex++) {
            double X0 = ex * hx;
            double Y0 = ey * hy;
            double xnode[4] = { X0, X0 + hx, X0 + hx, X0 };
            double ynode[4] = { Y0, Y0, Y0 + hy, Y0 + hy };

            double Ke[64] = {0.0};

            for (int gi = 0; gi < 2; gi++) {
                for (int gj = 0; gj < 2; gj++) {
                    double xi = gp[gi];
                    double eta = gp[gj];
                    double weight = gw[gi] * gw[gj];

                    double dN_dxi[4], dN_deta[4];

                    dN_dxi[0] = -0.25 * (1.0 - eta);
                    dN_dxi[1] =  0.25 * (1.0 - eta);
                    dN_dxi[2] =  0.25 * (1.0 + eta);
                    dN_dxi[3] = -0.25 * (1.0 + eta);

                    dN_deta[0] = -0.25 * (1.0 - xi);
                    dN_deta[1] = -0.25 * (1.0 + xi);
                    dN_deta[2] =  0.25 * (1.0 + xi);
                    dN_deta[3] =  0.25 * (1.0 - xi);

                    double J11 = 0.0, J12 = 0.0, J21 = 0.0, J22 = 0.0;
                    for (int i = 0; i < 4; i++) {
                        J11 += dN_dxi[i] * xnode[i];
                        J12 += dN_dxi[i] * ynode[i];
                        J21 += dN_deta[i] * xnode[i];
                        J22 += dN_deta[i] * ynode[i];
                    }

                    double detJ = J11 * J22 - J12 * J21;
                    double inv00 =  J22 / detJ;
                    double inv01 = -J12 / detJ;
                    double inv10 = -J21 / detJ;
                    double inv11 =  J11 / detJ;

                    double dNdx[4], dNdy[4];
                    for (int i = 0; i < 4; i++) {
                        dNdx[i] = inv00 * dN_dxi[i] + inv01 * dN_deta[i];
                        dNdy[i] = inv10 * dN_dxi[i] + inv11 * dN_deta[i];
                    }

                    double B[3][8] = {0.0};
                    for (int i = 0; i < 4; i++) {
                        B[0][2 * i]     = dNdx[i];
                        B[0][2 * i + 1] = 0.0;
                        B[1][2 * i]     = 0.0;
                        B[1][2 * i + 1] = dNdy[i];
                        B[2][2 * i]     = dNdy[i];
                        B[2][2 * i + 1] = dNdx[i];
                    }

                    for (int i = 0; i < 8; i++) {
                        for (int j = 0; j < 8; j++) {
                            double sum = 0.0;
                            for (int m = 0; m < 3; m++) {
                                for (int n = 0; n < 3; n++) {
                                    sum += B[m][i] * D[m][n] * B[n][j];
                                }
                            }
                            Ke[i * 8 + j] += sum * detJ * weight;
                        }
                    }
                }
            }

            int gdof[8];
            for (int li = 0; li < 4; li++) {
                int lx = (li == 1 || li == 2) ? 1 : 0;
                int ly = (li == 2 || li == 3) ? 1 : 0;
                int node = (ey + ly) * (nx + 1) + (ex + lx);
                gdof[2 * li]     = 2 * node;
                gdof[2 * li + 1] = 2 * node + 1;
            }

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    K[(size_t)gdof[i] * (size_t)n_dof + (size_t)gdof[j]] += Ke[i * 8 + j];
                }
            }
        }
    }

    for (int iy = 0; iy <= ny; iy++) {
        int node = iy * (nx + 1) + nx;
        int dof = 2 * node + 1;
        F[dof] += total_y / (double)(ny + 1);
    }

    int nfixed = (ny + 1) * 2;
    int *fixed = malloc((size_t)nfixed * sizeof(int));
    if (!fixed) {
        free(K);
        free(F);
        free(u);
        return 1;
    }

    int idx = 0;
    for (int iy = 0; iy <= ny; iy++) {
        int node = iy * (nx + 1);
        fixed[idx++] = 2 * node;
        fixed[idx++] = 2 * node + 1;
    }

    for (int k = 0; k < nfixed; k++) {
        int d = fixed[k];
        for (int j = 0; j < n_dof; j++) {
            K[(size_t)d * (size_t)n_dof + (size_t)j] = 0.0;
        }
        for (int i = 0; i < n_dof; i++) {
            K[(size_t)i * (size_t)n_dof + (size_t)d] = 0.0;
        }
    }

    for (int k = 0; k < nfixed; k++) {
        int d = fixed[k];
        K[(size_t)d * (size_t)n_dof + (size_t)d] = 1.0;
        F[d] = 0.0;
    }
    free(fixed);

    for (int k = 0; k < n_dof; k++) {
        int p = k;
        double maxv = fabs(K[(size_t)k * (size_t)n_dof + (size_t)k]);
        for (int i = k + 1; i < n_dof; i++) {
            double v = fabs(K[(size_t)i * (size_t)n_dof + (size_t)k]);
            if (v > maxv) {
                maxv = v;
                p = i;
            }
        }
        if (maxv < 1e-18) {
            free(K);
            free(F);
            free(u);
            return 1;
        }
        if (p != k) {
            for (int j = k; j < n_dof; j++) {
                double tmp = K[(size_t)k * (size_t)n_dof + (size_t)j];
                K[(size_t)k * (size_t)n_dof + (size_t)j] = K[(size_t)p * (size_t)n_dof + (size_t)j];
                K[(size_t)p * (size_t)n_dof + (size_t)j] = tmp;
            }
            double tmpb = F[k];
            F[k] = F[p];
            F[p] = tmpb;
        }

        double piv = K[(size_t)k * (size_t)n_dof + (size_t)k];
        for (int i = k + 1; i < n_dof; i++) {
            double factor = K[(size_t)i * (size_t)n_dof + (size_t)k] / piv;
            if (factor != 0.0) {
                K[(size_t)i * (size_t)n_dof + (size_t)k] = 0.0;
                for (int j = k + 1; j < n_dof; j++) {
                    K[(size_t)i * (size_t)n_dof + (size_t)j] -=
                        factor * K[(size_t)k * (size_t)n_dof + (size_t)j];
                }
                F[i] -= factor * F[k];
            }
        }
    }

    for (int i = n_dof - 1; i >= 0; i--) {
        double sum = F[i];
        for (int j = i + 1; j < n_dof; j++) {
            sum -= K[(size_t)i * (size_t)n_dof + (size_t)j] * u[j];
        }
        u[i] = sum / K[(size_t)i * (size_t)n_dof + (size_t)i];
    }

    double sum_uy = 0.0;
    for (int iy = 0; iy <= ny; iy++) {
        int node = iy * (nx + 1) + nx;
        int dof = 2 * node + 1;
        sum_uy += u[dof];
    }

    double avg_uy = sum_uy / (double)(ny + 1);
    double tip_uy = fabs(avg_uy);
    printf("tip_uy=%.10g\n", tip_uy);

    free(K);
    free(F);
    free(u);
    return 0;
}