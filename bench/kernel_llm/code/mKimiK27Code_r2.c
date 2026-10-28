#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double *alloc_d(size_t n) {
    return (double *)calloc(n, sizeof(double));
}

static void compute_D(double E, double nu, double D[3][3]) {
    double factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu));
    D[0][0] = factor * (1.0 - nu);
    D[0][1] = factor * nu;
    D[0][2] = 0.0;
    D[1][0] = factor * nu;
    D[1][1] = factor * (1.0 - nu);
    D[1][2] = 0.0;
    D[2][0] = 0.0;
    D[2][1] = 0.0;
    D[2][2] = factor * (0.5 - nu);
}

int main(int argc, char *argv[]) {
    if (argc < 6) return 1;
    double L, H, total_y;
    int nx, ny;
    L = atof(argv[1]);
    H = atof(argv[2]);
    nx = atoi(argv[3]);
    ny = atoi(argv[4]);
    total_y = atof(argv[5]);

    if (nx < 2 || ny < 2) return 1;

    const double E = 210000.0;
    const double nu = 0.3;
    double D[3][3];
    compute_D(E, nu, D);

    int nx_nodes = nx;
    int ny_nodes = ny;
    int num_nodes = nx_nodes * ny_nodes;
    int num_elems = (nx_nodes - 1) * (ny_nodes - 1);
    int dof = 2 * num_nodes;

    double *x = alloc_d(num_nodes);
    double *y = alloc_d(num_nodes);
    double dx = L / (nx_nodes - 1);
    double dy = H / (ny_nodes - 1);

    for (int j = 0; j < ny_nodes; j++) {
        for (int i = 0; i < nx_nodes; i++) {
            int node = j * nx_nodes + i;
            x[node] = i * dx;
            y[node] = j * dy;
        }
    }

    double *K = alloc_d(dof * dof);
    double *F = alloc_d(dof);

    int *elem_nodes = (int *)malloc(4 * num_elems * sizeof(int));
    for (int j = 0; j < ny_nodes - 1; j++) {
        for (int i = 0; i < nx_nodes - 1; i++) {
            int e = j * (nx_nodes - 1) + i;
            elem_nodes[4 * e + 0] = j * nx_nodes + i;
            elem_nodes[4 * e + 1] = j * nx_nodes + (i + 1);
            elem_nodes[4 * e + 2] = (j + 1) * nx_nodes + (i + 1);
            elem_nodes[4 * e + 3] = (j + 1) * nx_nodes + i;
        }
    }

    double gauss = 1.0 / sqrt(3.0);
    double xi[4] = {-gauss, gauss, gauss, -gauss};
    double eta[4] = {-gauss, -gauss, gauss, gauss};

    for (int e = 0; e < num_elems; e++) {
        int n[4];
        for (int k = 0; k < 4; k++) n[k] = elem_nodes[4 * e + k];
        double xe[4], ye[4];
        for (int k = 0; k < 4; k++) {
            xe[k] = x[n[k]];
            ye[k] = y[n[k]];
        }

        double ke[8][8] = {0};

        for (int gp = 0; gp < 4; gp++) {
            double s = xi[gp], t = eta[gp];
            double N1s = -0.25 * (1.0 - t);
            double N2s = 0.25 * (1.0 - t);
            double N3s = 0.25 * (1.0 + t);
            double N4s = -0.25 * (1.0 + t);
            double N1t = -0.25 * (1.0 - s);
            double N2t = -0.25 * (1.0 + s);
            double N3t = 0.25 * (1.0 + s);
            double N4t = 0.25 * (1.0 - s);

            double xs = N1s * xe[0] + N2s * xe[1] + N3s * xe[2] + N4s * xe[3];
            double xt = N1t * xe[0] + N2t * xe[1] + N3t * xe[2] + N4t * xe[3];
            double ys = N1s * ye[0] + N2s * ye[1] + N3s * ye[2] + N4s * ye[3];
            double yt = N1t * ye[0] + N2t * ye[1] + N3t * ye[2] + N4t * ye[3];

            double J = xs * yt - xt * ys;
            double invJ = 1.0 / J;

            double Nx[4], Ny[4];
            Nx[0] = invJ * (yt * N1s - ys * N1t);
            Ny[0] = invJ * (-xt * N1s + xs * N1t);
            Nx[1] = invJ * (yt * N2s - ys * N2t);
            Ny[1] = invJ * (-xt * N2s + xs * N2t);
            Nx[2] = invJ * (yt * N3s - ys * N3t);
            Ny[2] = invJ * (-xt * N3s + xs * N3t);
            Nx[3] = invJ * (yt * N4s - ys * N4t);
            Ny[3] = invJ * (-xt * N4s + xs * N4t);

            double B[3][8];
            for (int k = 0; k < 4; k++) {
                B[0][2 * k] = Nx[k];
                B[0][2 * k + 1] = 0.0;
                B[1][2 * k] = 0.0;
                B[1][2 * k + 1] = Ny[k];
                B[2][2 * k] = Ny[k];
                B[2][2 * k + 1] = Nx[k];
            }

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    double sum = 0.0;
                    for (int a = 0; a < 3; a++) {
                        for (int b = 0; b < 3; b++) {
                            sum += B[a][i] * D[a][b] * B[b][j];
                        }
                    }
                    ke[i][j] += sum * J;
                }
            }
        }

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int gi = 2 * n[i];
                int gj = 2 * n[j];
                K[(gi) * dof + (gj)] += ke[2 * i][2 * j];
                K[(gi) * dof + (gj + 1)] += ke[2 * i][2 * j + 1];
                K[(gi + 1) * dof + (gj)] += ke[2 * i + 1][2 * j];
                K[(gi + 1) * dof + (gj + 1)] += ke[2 * i + 1][2 * j + 1];
            }
        }
    }

    int *bc_nodes = (int *)malloc(num_nodes * sizeof(int));
    int nbc = 0;
    for (int j = 0; j < ny_nodes; j++) {
        for (int i = 0; i < nx_nodes; i++) {
            if (fabs(x[j * nx_nodes + i]) < 1e-12) {
                bc_nodes[nbc++] = j * nx_nodes + i;
            }
        }
    }

    int *is_fixed = (int *)calloc(num_nodes, sizeof(int));
    for (int i = 0; i < nbc; i++) is_fixed[bc_nodes[i]] = 1;

    int *tip_nodes = (int *)malloc(ny_nodes * sizeof(int));
    int ntip = 0;
    for (int j = 0; j < ny_nodes; j++) {
        int node = j * nx_nodes + (nx_nodes - 1);
        tip_nodes[ntip++] = node;
    }

    double force_per_node = total_y / ntip;
    for (int i = 0; i < ntip; i++) {
        int node = tip_nodes[i];
        F[2 * node + 1] = force_per_node;
    }

    double *A = alloc_d(dof * (dof + 1));
    for (int i = 0; i < dof; i++) {
        for (int j = 0; j < dof; j++) {
            A[i * (dof + 1) + j] = K[i * dof + j];
        }
        A[i * (dof + 1) + dof] = F[i];
    }

    for (int i = 0; i < nbc; i++) {
        int node = bc_nodes[i];
        for (int d = 0; d < 2; d++) {
            int row = 2 * node + d;
            for (int j = 0; j < dof; j++) {
                A[row * (dof + 1) + j] = 0.0;
            }
            A[row * (dof + 1) + row] = 1.0;
            A[row * (dof + 1) + dof] = 0.0;
        }
    }

    for (int col = 0; col < dof; col++) {
        int pivot = col;
        double max_val = fabs(A[pivot * (dof + 1) + col]);
        for (int row = col + 1; row < dof; row++) {
            double val = fabs(A[row * (dof + 1) + col]);
            if (val > max_val) {
                max_val = val;
                pivot = row;
            }
        }
        if (pivot != col) {
            for (int j = 0; j < dof + 1; j++) {
                double tmp = A[col * (dof + 1) + j];
                A[col * (dof + 1) + j] = A[pivot * (dof + 1) + j];
                A[pivot * (dof + 1) + j] = tmp;
            }
        }
        double diag = A[col * (dof + 1) + col];
        if (fabs(diag) < 1e-15) continue;
        for (int j = col; j < dof + 1; j++) {
            A[col * (dof + 1) + j] /= diag;
        }
        for (int row = 0; row < dof; row++) {
            if (row == col) continue;
            double factor = A[row * (dof + 1) + col];
            if (fabs(factor) < 1e-15) continue;
            for (int j = col; j < dof + 1; j++) {
                A[row * (dof + 1) + j] -= factor * A[col * (dof + 1) + j];
            }
        }
    }

    double sum_uy = 0.0;
    for (int i = 0; i < ntip; i++) {
        int node = tip_nodes[i];
        sum_uy += A[(2 * node + 1) * (dof + 1) + dof];
    }
    double avg_uy = sum_uy / ntip;
    double result = fabs(avg_uy);

    printf("tip_uy=%.10g\n", result);

    free(x);
    free(y);
    free(K);
    free(F);
    free(elem_nodes);
    free(bc_nodes);
    free(is_fixed);
    free(tip_nodes);
    free(A);

    return 0;
}