#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct { double k[4][4]; } ElemK; /* local stiffness: [node][node] is 2x2 block; we store full 8x8 */

static void elem_stiffness(double E, double nu, double Le, double He,
                           double xe[4], double ye[4], double ke[8][8])
{
    double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double mu = E / (2.0 * (1.0 + nu));
    double gp = 1.0 / sqrt(3.0), g[2] = { -gp, gp }, w = 1.0;
    int i, j;
    for (i = 0; i < 8; i++)
        for (j = 0; j < 8; j++) ke[i][j] = 0.0;

    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            double xi = g[i], eta = g[j];
            /* shape function derivatives wrt xi, eta */
            double dNxi[4] = { -(1 - eta), (1 - eta), (1 + eta), -(1 + eta) };
            double dNeta[4] = { -(1 - xi), -(1 + xi), (1 + xi), (1 - xi) };
            double J[2][2], Jinv[2][2], detJ;
            double dNdx[4], dNdy[4];
            double B[3][8];
            double D[3][3];
            int a, b, p, q;

            J[0][0] = J[0][1] = J[1][0] = J[1][1] = 0.0;
            for (a = 0; a < 4; a++) {
                J[0][0] += dNxi[a] * xe[a];
                J[0][1] += dNxi[a] * ye[a];
                J[1][0] += dNeta[a] * xe[a];
                J[1][1] += dNeta[a] * ye[a];
            }
            detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0];
            Jinv[0][0] = J[1][1] / detJ;
            Jinv[0][1] = -J[0][1] / detJ;
            Jinv[1][0] = -J[1][0] / detJ;
            Jinv[1][1] = J[0][0] / detJ;
            for (a = 0; a < 4; a++) {
                dNdx[a] = Jinv[0][0] * dNxi[a] + Jinv[0][1] * dNeta[a];
                dNdy[a] = Jinv[1][0] * dNxi[a] + Jinv[1][1] * dNeta[a];
            }

            for (p = 0; p < 3; p++)
                for (q = 0; q < 8; q++) B[p][q] = 0.0;
            for (a = 0; a < 4; a++) {
                B[0][2 * a] = dNdx[a];
                B[1][2 * a + 1] = dNdy[a];
                B[2][2 * a] = dNdy[a];
                B[2][2 * a + 1] = dNdx[a];
            }

            D[0][0] = lambda + 2.0 * mu;
            D[0][1] = lambda;
            D[0][2] = 0.0;
            D[1][0] = lambda;
            D[1][1] = lambda + 2.0 * mu;
            D[1][2] = 0.0;
            D[2][0] = D[2][1] = 0.0;
            D[2][2] = mu;

            /* ke += B^T D B * detJ * w^2 */
            {
                double DB[3][8];
                for (p = 0; p < 3; p++)
                    for (q = 0; q < 8; q++) {
                        double s = 0.0;
                        for (a = 0; a < 3; a++) s += D[p][a] * B[a][q];
                        DB[p][q] = s;
                    }
                for (a = 0; a < 8; a++)
                    for (b = 0; b < 8; b++) {
                        double s = 0.0;
                        for (p = 0; p < 3; p++) s += B[p][a] * DB[p][b];
                        ke[a][b] += s * detJ * w * w;
                    }
            }
        }
    }
}

static void solve_gauss(double **A, double *b, int n)
{
    int i, j, k, piv;
    for (k = 0; k < n; k++) {
        double mx = fabs(A[k][k]);
        piv = k;
        for (i = k + 1; i < n; i++)
            if (fabs(A[i][k]) > mx) { mx = fabs(A[i][k]); piv = i; }
        if (piv != k) {
            double *tmp = A[k]; A[k] = A[piv]; A[piv] = tmp;
            { double t = b[k]; b[k] = b[piv]; b[piv] = t; }
        }
        for (i = k + 1; i < n; i++) {
            double f = A[i][k] / A[k][k];
            if (f != 0.0) {
                for (j = k; j < n; j++) A[i][j] -= f * A[k][j];
                b[i] -= f * b[k];
            }
        }
    }
    for (i = n - 1; i >= 0; i--) {
        double s = b[i];
        for (j = i + 1; j < n; j++) s -= A[i][j] * b[j];
        b[i] = s / A[i][i];
    }
}

int main(int argc, char **argv)
{
    double L, H, total_y, E = 210000.0, nu = 0.3;
    int nx, ny, i, j, e, ndof, nn;
    double dx, dy;
    double *x, *y;
    int **conn;           /* conn[e][4] */
    int *bc;              /* 1 = fixed */
    double **K, *f, *u;
    double sum = 0.0;
    int n_right, cnt = 0;

    if (argc < 6) return 1;
    L = atof(argv[1]);
    H = atof(argv[2]);
    nx = atoi(argv[3]);
    ny = atoi(argv[4]);
    total_y = atof(argv[5]);
    if (nx < 1 || ny < 1) return 1;

    nn = (nx + 1) * (ny + 1);
    ndof = 2 * nn;
    dx = L / nx;
    dy = H / ny;

    x = malloc(nn * sizeof(double));
    y = malloc(nn * sizeof(double));
    for (j = 0; j <= ny; j++)
        for (i = 0; i <= nx; i++) {
            int n = j * (nx + 1) + i;
            x[n] = i * dx;
            y[n] = j * dy;
        }

    conn = malloc(nx * ny * sizeof(int *));
    for (e = 0; e < nx * ny; e++) {
        int iex = e % nx, iey = e / nx;
        conn[e] = malloc(4 * sizeof(int));
        conn[e][0] = iey * (nx + 1) + iex;
        conn[e][1] = iey * (nx + 1) + iex + 1;
        conn[e][2] = (iey + 1) * (nx + 1) + iex + 1;
        conn[e][3] = (iey + 1) * (nx + 1) + iex;
    }

    bc = calloc(nn, sizeof(int));
    for (j = 0; j <= ny; j++) bc[j * (nx + 1)] = 1;

    K = malloc(ndof * sizeof(double *));
    for (i = 0; i < ndof; i++) K[i] = calloc(ndof, sizeof(double));
    f = calloc(ndof, sizeof(double));
    u = calloc(ndof, sizeof(double));

    for (e = 0; e < nx * ny; e++) {
        double xe[4], ye[4], ke[8][8];
        int a, b;
        for (a = 0; a < 4; a++) { xe[a] = x[conn[e][a]]; ye[a] = y[conn[e][a]]; }
        elem_stiffness(E, nu, dx, dy, xe, ye, ke);
        for (a = 0; a < 4; a++)
            for (b = 0; b < 4; b++) {
                K[2 * conn[e][a]][2 * conn[e][b]] += ke[2 * a][2 * b];
                K[2 * conn[e][a]][2 * conn[e][b] + 1] += ke[2 * a][2 * b + 1];
                K[2 * conn[e][a] + 1][2 * conn[e][b]] += ke[2 * a + 1][2 * b];
                K[2 * conn[e][a] + 1][2 * conn[e][b] + 1] += ke[2 * a + 1][2 * b + 1];
            }
    }

    /* loads on x=L nodes (i = nx), uy dof */
    n_right = ny + 1;
    for (j = 0; j <= ny; j++)
        f[2 * (j * (nx + 1) + nx) + 1] = total_y / n_right;

    /* Dirichlet: zero rows/cols, put 1 on diagonal, RHS 0 */
    for (j = 0; j <= ny; j++) {
        int n = j * (nx + 1);
        int d;
        for (d = 0; d < 2; d++) {
            int dof = 2 * n + d;
            int k;
            for (k = 0; k < ndof; k++) { K[k][dof] = 0.0; K[dof][k] = 0.0; }
            K[dof][dof] = 1.0;
            f[dof] = 0.0;
        }
    }

    solve_gauss(K, f, ndof);

    for (j = 0; j <= ny; j++) {
        int n = j * (nx + 1) + nx;
        sum += f[2 * n + 1];
        cnt++;
    }

    printf("tip_uy=%.10g\n", fabs(sum / cnt));
    return 0;
}