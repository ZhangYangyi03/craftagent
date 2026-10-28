#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int NX, NY, NNODE, NDOF;
static double L, H, TOTALY;

static int nid(int i, int j) { return j * (NX + 1) + i; } /* i: x index, j: y index */

int main(int argc, char **argv)
{
    if (argc < 6) { fprintf(stderr, "usage: %s L H nx ny total_y\n", argv[0]); return 1; }
    L = atof(argv[1]);
    H = atof(argv[2]);
    NX = atoi(argv[3]);
    NY = atoi(argv[4]);
    TOTALY = atof(argv[5]);

    NNODE = (NX + 1) * (NY + 1);
    NDOF = 2 * NNODE;

    const double E = 210000.0, nu = 0.3;
    double lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double mu = E / (2.0 * (1.0 + nu));
    double D[3][3] = {
        { lam + 2.0 * mu, lam, 0.0 },
        { lam, lam + 2.0 * mu, 0.0 },
        { 0.0, 0.0, mu }
    };

    double *K = calloc((size_t)NDOF * NDOF, sizeof(double));
    double *F = calloc(NDOF, sizeof(double));
    double *U = calloc(NDOF, sizeof(double));
    if (!K || !F || !U) { fprintf(stderr, "alloc fail\n"); return 1; }

    double hx = L / NX, hy = H / NY;
    const double g = 1.0 / sqrt(3.0);
    double gp[2] = { -g, g };

    for (int ej = 0; ej < NY; ej++) {
        for (int ei = 0; ei < NX; ei++) {
            int en[4] = { nid(ei, ej), nid(ei + 1, ej), nid(ei + 1, ej + 1), nid(ei, ej + 1) };
            double ke[8][8] = {{0}};
            for (int a = 0; a < 2; a++) {
                for (int b = 0; b < 2; b++) {
                    double xi = gp[a], eta = gp[b];
                    /* shape functions and derivatives */
                    double N[4] = {
                        0.25 * (1 - xi) * (1 - eta),
                        0.25 * (1 + xi) * (1 - eta),
                        0.25 * (1 + xi) * (1 + eta),
                        0.25 * (1 - xi) * (1 + eta)
                    };
                    double dN[4][2] = {
                        { -0.25 * (1 - eta), -0.25 * (1 - xi) },
                        {  0.25 * (1 - eta), -0.25 * (1 + xi) },
                        {  0.25 * (1 + eta),  0.25 * (1 + xi) },
                        { -0.25 * (1 + eta),  0.25 * (1 - xi) }
                    };
                    double J[2][2] = {{0}};
                    for (int n = 0; n < 4; n++) {
                        double x = (en[n] % (NX + 1)) * hx;
                        double y = (en[n] / (NX + 1)) * hy;
                        J[0][0] += dN[n][0] * x; J[0][1] += dN[n][0] * y;
                        J[1][0] += dN[n][1] * x; J[1][1] += dN[n][1] * y;
                    }
                    double detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0];
                    double detJinv = 1.0 / detJ;
                    double B[3][8] = {{0}};
                    for (int n = 0; n < 4; n++) {
                        double dx =  detJinv * ( J[1][1] * dN[n][0] - J[0][1] * dN[n][1]);
                        double dy =  detJinv * (-J[1][0] * dN[n][0] + J[0][0] * dN[n][1]);
                        B[0][2 * n]     = dx;
                        B[1][2 * n + 1] = dy;
                        B[2][2 * n]     = dy;
                        B[2][2 * n + 1] = dx;
                    }
                    double DB[3][8];
                    for (int r = 0; r < 3; r++)
                        for (int c = 0; c < 8; c++) {
                            double s = 0.0;
                            for (int m = 0; m < 3; m++) s += D[r][m] * B[m][c];
                            DB[r][c] = s;
                        }
                    double w = detJ; /* weights = 1 */
                    for (int r = 0; r < 8; r++)
                        for (int c = 0; c < 8; c++) {
                            double s = 0.0;
                            for (int m = 0; m < 3; m++) s += B[m][r] * DB[m][c];
                            ke[r][c] += w * s;
                        }
                }
            }
            int edof[8];
            for (int n = 0; n < 4; n++) { edof[2 * n] = 2 * en[n]; edof[2 * n + 1] = 2 * en[n] + 1; }
            for (int r = 0; r < 8; r++)
                for (int c = 0; c < 8; c++)
                    K[(size_t)edof[r] * NDOF + edof[c]] += ke[r][c];
        }
    }

    /* loads on x = L edge */
    int nedge = NY + 1;
    double fy = TOTALY / (double)nedge;
    for (int j = 0; j <= NY; j++) {
        int n = nid(NX, j);
        F[2 * n + 1] += fy;
    }

    /* Dirichlet: x = 0 edge fixed */
    for (int j = 0; j <= NY; j++) {
        int n = nid(0, j);
        int d[2] = { 2 * n, 2 * n + 1 };
        for (int k = 0; k < 2; k++) {
            int dk = d[k];
            for (int c = 0; c < NDOF; c++) K[(size_t)dk * NDOF + c] = 0.0;
            for (int r = 0; r < NDOF; r++) K[(size_t)r * NDOF + dk] = 0.0;
            K[(size_t)dk * NDOF + dk] = 1.0;
            F[dk] = 0.0;
        }
    }

    /* dense Gaussian elimination with partial pivoting */
    for (int col = 0; col < NDOF; col++) {
        int piv = col;
        double best = fabs(K[(size_t)col * NDOF + col]);
        for (int r = col + 1; r < NDOF; r++) {
            double v = fabs(K[(size_t)r * NDOF + col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best == 0.0) continue;
        if (piv != col) {
            for (int c = 0; c < NDOF; c++) {
                double t = K[(size_t)col * NDOF + c];
                K[(size_t)col * NDOF + c] = K[(size_t)piv * NDOF + c];
                K[(size_t)piv * NDOF + c] = t;
            }
            double t = F[col]; F[col] = F[piv]; F[piv] = t;
        }
        double d = K[(size_t)col * NDOF + col];
        for (int r = col + 1; r < NDOF; r++) {
            double m = K[(size_t)r * NDOF + col] / d;
            if (m == 0.0) continue;
            K[(size_t)r * NDOF + col] = 0.0;
            for (int c = col + 1; c < NDOF; c++)
                K[(size_t)r * NDOF + c] -= m * K[(size_t)col * NDOF + c];
            F[r] -= m * F[col];
        }
    }
    for (int r = NDOF - 1; r >= 0; r--) {
        double s = F[r];
        for (int c = r + 1; c < NDOF; c++) s -= K[(size_t)r * NDOF + c] * U[c];
        double d = K[(size_t)r * NDOF + r];
        U[r] = (d != 0.0) ? s / d : 0.0;
    }

    double sum = 0.0;
    for (int j = 0; j <= NY; j++) sum += U[2 * nid(NX, j) + 1];
    double tip = fabs(sum / (double)nedge);

    printf("tip_uy=%.10g\n", tip);

    free(K); free(F); free(U);
    return 0;
}