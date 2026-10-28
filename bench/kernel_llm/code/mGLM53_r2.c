#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static double E_ = 210000.0, nu_ = 0.3;

int main(int argc, char **argv)
{
    if (argc < 6) { fprintf(stderr, "usage: %s L H nx ny total_y\n", argv[0]); return 1; }
    double L = atof(argv[1]);
    double H = atof(argv[2]);
    int nx = atoi(argv[3]);
    int ny = atoi(argv[4]);
    double total_y = atof(argv[5]);

    int npx = nx + 1, npy = ny + 1;
    long nn = (long)npx * npy;
    long ndof = 2 * nn;

    /* coordinates */
    double *X = malloc(sizeof(double) * nn);
    double *Y = malloc(sizeof(double) * nn);
    for (int j = 0; j < npy; j++)
        for (int i = 0; i < npx; i++) {
            long n = (long)j * npx + i;
            X[n] = L * (double)i / nx;
            Y[n] = H * (double)j / ny;
        }

    /* plane strain D matrix */
    double c = E_ / ((1.0 + nu_) * (1.0 - 2.0 * nu_));
    double D[3][3];
    D[0][0] = c * (1.0 - nu_); D[0][1] = c * nu_;        D[0][2] = 0.0;
    D[1][0] = c * nu_;        D[1][1] = c * (1.0 - nu_); D[1][2] = 0.0;
    D[2][0] = 0.0;             D[2][1] = 0.0;             D[2][2] = c * 0.5 * (1.0 - 2.0 * nu_);

    double *K = calloc((size_t)ndof * (size_t)ndof, sizeof(double));
    double *F = calloc((size_t)ndof, sizeof(double));
    if (!K || !F) { fprintf(stderr, "alloc fail\n"); return 1; }

    double gx[2] = { -1.0 / sqrt(3.0), 1.0 / sqrt(3.0) };

    for (int ej = 0; ej < ny; ej++)
        for (int ei = 0; ei < nx; ei++) {
            long nodes[4];
            nodes[0] = (long)ej * npx + ei;
            nodes[1] = (long)ej * npx + ei + 1;
            nodes[2] = (long)(ej + 1) * npx + ei + 1;
            nodes[3] = (long)(ej + 1) * npx + ei;
            long dofs[8];
            for (int a = 0; a < 4; a++) { dofs[2*a] = 2*nodes[a]; dofs[2*a+1] = 2*nodes[a]+1; }

            double ke[8][8] = {{0}};
            for (int qi = 0; qi < 2; qi++)
                for (int qj = 0; qj < 2; qj++) {
                    double xi = gx[qi], eta = gx[qj];
                    double dN[4][2] = {
                        { -0.25*(1.0-eta), -0.25*(1.0-xi) },
                        {  0.25*(1.0-eta), -0.25*(1.0+xi) },
                        {  0.25*(1.0+eta),  0.25*(1.0+xi) },
                        { -0.25*(1.0+eta),  0.25*(1.0-xi) }
                    };
                    double J[2][2] = {{0}};
                    for (int a = 0; a < 4; a++) {
                        J[0][0] += dN[a][0] * X[nodes[a]];
                        J[0][1] += dN[a][0] * Y[nodes[a]];
                        J[1][0] += dN[a][1] * X[nodes[a]];
                        J[1][1] += dN[a][1] * Y[nodes[a]];
                    }
                    double detJ = J[0][0]*J[1][1] - J[0][1]*J[1][0];
                    double invJ[2][2] = { { J[1][1]/detJ, -J[0][1]/detJ },
                                          { -J[1][0]/detJ, J[0][0]/detJ } };
                    double B[3][8] = {{0}};
                    for (int a = 0; a < 4; a++) {
                        double dNx = invJ[0][0]*dN[a][0] + invJ[0][1]*dN[a][1];
                        double dNy = invJ[1][0]*dN[a][0] + invJ[1][1]*dN[a][1];
                        B[0][2*a]   = dNx;
                        B[1][2*a+1] = dNy;
                        B[2][2*a]   = dNy;
                        B[2][2*a+1] = dNx;
                    }
                    double DB[3][8];
                    for (int r = 0; r < 3; r++)
                        for (int s = 0; s < 8; s++) {
                            double s2 = 0.0;
                            for (int m = 0; m < 3; m++) s2 += D[r][m] * B[m][s];
                            DB[r][s] = s2;
                        }
                    for (int r = 0; r < 8; r++)
                        for (int s = 0; s < 8; s++) {
                            double s2 = 0.0;
                            for (int m = 0; m < 3; m++) s2 += B[m][r] * DB[m][s];
                            ke[r][s] += s2 * detJ; /* weights = 1 */
                        }
                }
            for (int r = 0; r < 8; r++)
                for (int s = 0; s < 8; s++)
                    K[dofs[r]*ndof + dofs[s]] += ke[r][s];
        }

    /* load on x = L edge */
    double fy = total_y / (double)npy;
    for (int j = 0; j < npy; j++) {
        long n = (long)j * npx + nx;
        F[2*n + 1] += fy;
    }

    /* Dirichlet: x = 0 edge, ux = uy = 0 */
    for (int j = 0; j < npy; j++) {
        long n = (long)j * npx + 0;
        for (int c = 0; c < 2; c++) {
            long d = 2*n + c;
            for (long s = 0; s < ndof; s++) K[d*ndof + s] = 0.0;
            for (long r = 0; r < ndof; r++) K[r*ndof + d] = 0.0;
            K[d*ndof + d] = 1.0;
            F[d] = 0.0;
        }
    }

    /* dense Gaussian elimination with partial pivoting */
    for (long i = 0; i < ndof; i++) {
        long p = i;
        double mx = fabs(K[i*ndof + i]);
        for (long r = i + 1; r < ndof; r++) {
            double v = fabs(K[r*ndof + i]);
            if (v > mx) { mx = v; p = r; }
        }
        if (mx == 0.0) continue;
        if (p != i) {
            for (long s = 0; s < ndof; s++) {
                double t = K[i*ndof + s]; K[i*ndof + s] = K[p*ndof + s]; K[p*ndof + s] = t;
            }
            double t = F[i]; F[i] = F[p]; F[p] = t;
        }
        double piv = K[i*ndof + i];
        for (long r = i + 1; r < ndof; r++) {
            double m = K[r*ndof + i] / piv;
            if (m == 0.0) continue;
            for (long s = i; s < ndof; s++) K[r*ndof + s] -= m * K[i*ndof + s];
            F[r] -= m * F[i];
        }
    }
    double *U = malloc(sizeof(double) * ndof);
    for (long i = ndof - 1; i >= 0; i--) {
        double s2 = F[i];
        for (long j2 = i + 1; j2 < ndof; j2++) s2 -= K[i*ndof + j2] * U[j2];
        U[i] = (K[i*ndof + i] != 0.0) ? s2 / K[i*ndof + i] : 0.0;
    }

    /* average uy on x = L edge */
    double sum = 0.0;
    for (int j = 0; j < npy; j++) {
        long n = (long)j * npx + nx;
        sum += U[2*n + 1];
    }
    double avg = sum / (double)npy;

    printf("tip_uy=%.10g\n", fabs(avg));

    free(X); free(Y); free(K); free(F); free(U);
    return 0;
}