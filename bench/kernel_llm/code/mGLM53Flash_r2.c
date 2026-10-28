#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Solve dense linear system A x = b with partial pivoting Gaussian elimination.
   A is n x n row-major, modified in place; b length n. */
static void gauss_solve(double *A, double *b, int n) {
    for (int k = 0; k < n; k++) {
        int piv = k;
        double mx = fabs(A[k*n + k]);
        for (int i = k+1; i < n; i++) {
            double v = fabs(A[i*n + k]);
            if (v > mx) { mx = v; piv = i; }
        }
        if (piv != k) {
            for (int j = k; j < n; j++) {
                double t = A[k*n + j];
                A[k*n + j] = A[piv*n + j];
                A[piv*n + j] = t;
            }
            double t = b[k]; b[k] = b[piv]; b[piv] = t;
        }
        double d = A[k*n + k];
        for (int i = k+1; i < n; i++) {
            double f = A[i*n + k] / d;
            if (f == 0.0) continue;
            for (int j = k; j < n; j++)
                A[i*n + j] -= f * A[k*n + j];
            b[i] -= f * b[k];
        }
    }
    for (int k = n-1; k >= 0; k--) {
        double s = b[k];
        for (int j = k+1; j < n; j++)
            s -= A[k*n + j] * b[j];
        b[k] = s / A[k*n + k];
    }
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s L H nx ny total_y\n", argv[0]); return 1; }
    double L = atof(argv[1]);
    double H = atof(argv[2]);
    int nx = atoi(argv[3]);
    int ny = atoi(argv[4]);
    double total_y = atof(argv[5]);

    const double E = 210000.0, nu = 0.3;

    int npx = nx + 1, npy = ny + 1;
    int nn = npx * npy;
    int ndof = 2 * nn;
    int ne = nx * ny;

    /* plane strain constitutive matrix D */
    double lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double mu  = E / (2.0 * (1.0 + nu));
    double D[3][3] = {
        { lam + 2.0*mu, lam,          0.0 },
        { lam,          lam + 2.0*mu, 0.0 },
        { 0.0,          0.0,          mu  }
    };

    double *K = calloc((size_t)ndof * ndof, sizeof(double));
    double *F = calloc((size_t)ndof, sizeof(double));
    if (!K || !F) return 1;

    double dx = L / nx, dy = H / ny;

    /* Gauss points and weights, 2x2 */
    const double gp[2] = { -1.0/sqrt(3.0), 1.0/sqrt(3.0) };
    const double gw = 1.0;

    double Ke[8*8];

    for (int ey = 0; ey < ny; ey++) {
        for (int ex = 0; ex < nx; ex++) {
            /* element nodes, counterclockwise: bl, br, tr, tl */
            int nodes[4];
            nodes[0] = ey * npx + ex;            /* (ex,ey)     */
            nodes[1] = ey * npx + ex + 1;        /* (ex+1,ey)   */
            nodes[2] = (ey+1) * npx + ex + 1;    /* (ex+1,ey+1) */
            nodes[3] = (ey+1) * npx + ex;        /* (ex,ey+1)   */

            double xe[4], ye[4];
            for (int a = 0; a < 4; a++) {
                int i = nodes[a] % npx, j = nodes[a] / npx;
                xe[a] = i * dx;
                ye[a] = j * dy;
            }

            for (int i = 0; i < 64; i++) Ke[i] = 0.0;

            for (int gy = 0; gy < 2; gy++) {
                for (int gx = 0; gx < 2; gx++) {
                    double xi = gp[gx], eta = gp[gy];

                    /* shape functions and derivatives */
                    double N[4] = {
                        0.25*(1.0-xi)*(1.0-eta), 0.25*(1.0+xi)*(1.0-eta),
                        0.25*(1.0+xi)*(1.0+eta), 0.25*(1.0-xi)*(1.0+eta)
                    };
                    double dNxi[4]  = { -0.25*(1.0-eta), 0.25*(1.0-eta),
                                         0.25*(1.0+eta), -0.25*(1.0+eta) };
                    double dNeta[4] = { -0.25*(1.0-xi), -0.25*(1.0+xi),
                                         0.25*(1.0+xi),  0.25*(1.0-xi) };

                    double J[2][2];
                    J[0][0] = J[0][1] = J[1][0] = J[1][1] = 0.0;
                    for (int a = 0; a < 4; a++) {
                        J[0][0] += dNxi[a]  * xe[a];
                        J[0][1] += dNxi[a]  * ye[a];
                        J[1][0] += dNeta[a] * xe[a];
                        J[1][1] += dNeta[a] * ye[a];
                    }
                    double detJ = J[0][0]*J[1][1] - J[0][1]*J[1][0];
                    double iJ[2][2] = {
                        {  J[1][1]/detJ, -J[0][1]/detJ },
                        { -J[1][0]/detJ,  J[0][0]/detJ }
                    };

                    double dNdx[4], dNdy[4];
                    for (int a = 0; a < 4; a++) {
                        dNdx[a] = iJ[0][0]*dNxi[a] + iJ[0][1]*dNeta[a];
                        dNdy[a] = iJ[1][0]*dNxi[a] + iJ[1][1]*dNeta[a];
                    }

                    double w = gw * gw * detJ;

                    /* B matrix 3 x 8 */
                    double B[3][8];
                    for (int a = 0; a < 4; a++) {
                        B[0][2*a]   = dNdx[a]; B[0][2*a+1] = 0.0;
                        B[1][2*a]   = 0.0;     B[1][2*a+1] = dNdy[a];
                        B[2][2*a]   = dNdy[a]; B[2][2*a+1] = dNdx[a];
                    }

                    for (int a = 0; a < 4; a++)
                        for (int b = 0; b < 4; b++)
                            for (int p = 0; p < 3; p++)
                                for (int q = 0; q < 3; q++)
                                    Ke[(2*a+p)*8 + (2*b+q)] +=
                                        B[p][2*a] * D[p][q] * B[q][2*b] * w;
                }
            }

            for (int a = 0; a < 4; a++)
                for (int b = 0; b < 4; b++)
                    for (int p = 0; p < 2; p++)
                        for (int q = 0; q < 2; q++)
                            K[(2*nodes[a]+p)*ndof + (2*nodes[b]+q)] +=
                                Ke[(2*a+p)*8 + (2*b+q)];
        }
    }

    /* loads on x = L edge nodes */
    int nload = npy;
    double fy = total_y / nload;
    for (int j = 0; j < npy; j++)
        F[2 * (j * npx + nx) + 1] = fy;

    /* Dirichlet BC: x = 0 edge nodes, ux = uy = 0 */
    for (int j = 0; j < npy; j++) {
        for (int d = 0; d < 2; d++) {
            int r = 2 * (j * npx + 0) + d;
            for (int c = 0; c < ndof; c++) K[r*ndof + c] = 0.0;
            for (int r2 = 0; r2 < ndof; r2++) K[r2*ndof + r] = 0.0;
            K[r*ndof + r] = 1.0;
            F[r] = 0.0;
        }
    }

    gauss_solve(K, F, ndof);

    /* average uy over x = L nodes */
    double sum = 0.0;
    for (int j = 0; j < npy; j++)
        sum += F[2 * (j * npx + nx) + 1];
    double avg = sum / nload;

    printf("tip_uy=%.10g\n", fabs(avg));

    free(K);
    free(F);
    return 0;
}