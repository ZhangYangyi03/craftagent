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

    int num_nodes = (nx + 1) * (ny + 1);
    int num_dofs = 2 * num_nodes;
    double dx = L / nx;
    double dy = H / ny;

    double E = 210000.0;
    double nu = 0.3;
    double factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double D[3][3] = {
        {factor * (1.0 - nu), factor * nu, 0.0},
        {factor * nu, factor * (1.0 - nu), 0.0},
        {0.0, 0.0, factor * (1.0 - 2.0 * nu) / 2.0}
    };

    double *K = (double *)calloc((size_t)num_dofs * num_dofs, sizeof(double));
    double *F = (double *)calloc(num_dofs, sizeof(double));

    double gp = 1.0 / sqrt(3.0);
    double gauss_pts[2] = {-gp, gp};
    double gauss_wts[2] = {1.0, 1.0};

    for (int ey = 0; ey < ny; ey++) {
        for (int ex = 0; ex < nx; ex++) {
            int n[4];
            n[0] = ey * (nx + 1) + ex;
            n[1] = ey * (nx + 1) + ex + 1;
            n[2] = (ey + 1) * (nx + 1) + ex + 1;
            n[3] = (ey + 1) * (nx + 1) + ex;

            int dofs[8];
            for (int k = 0; k < 4; k++) {
                dofs[2 * k] = 2 * n[k];
                dofs[2 * k + 1] = 2 * n[k] + 1;
            }

            double Ke[8][8];
            memset(Ke, 0, sizeof(Ke));

            double x_coords[4] = {ex * dx, (ex + 1) * dx, (ex + 1) * dx, ex * dx};
            double y_coords[4] = {ey * dy, ey * dy, (ey + 1) * dy, (ey + 1) * dy};

            for (int gi = 0; gi < 2; gi++) {
                for (int gj = 0; gj < 2; gj++) {
                    double xi = gauss_pts[gi];
                    double eta = gauss_pts[gj];
                    double w = gauss_wts[gi] * gauss_wts[gj];

                    double dNdxi[4], dNdeta[4];
                    dNdxi[0] = -(1.0 - eta) / 4.0;
                    dNdxi[1] =  (1.0 - eta) / 4.0;
                    dNdxi[2] =  (1.0 + eta) / 4.0;
                    dNdxi[3] = -(1.0 + eta) / 4.0;

                    dNdeta[0] = -(1.0 - xi) / 4.0;
                    dNdeta[1] = -(1.0 + xi) / 4.0;
                    dNdeta[2] =  (1.0 + xi) / 4.0;
                    dNdeta[3] =  (1.0 - xi) / 4.0;

                    double J11 = 0.0, J12 = 0.0, J21 = 0.0, J22 = 0.0;
                    for (int k = 0; k < 4; k++) {
                        J11 += dNdxi[k] * x_coords[k];
                        J12 += dNdxi[k] * y_coords[k];
                        J21 += dNdeta[k] * x_coords[k];
                        J22 += dNdeta[k] * y_coords[k];
                    }
                    double detJ = J11 * J22 - J12 * J21;
                    double invJ11 =  J22 / detJ;
                    double invJ12 = -J12 / detJ;
                    double invJ21 = -J21 / detJ;
                    double invJ22 =  J11 / detJ;

                    double dNdx[4], dNdy[4];
                    for (int k = 0; k < 4; k++) {
                        dNdx[k] = invJ11 * dNdxi[k] + invJ12 * dNdeta[k];
                        dNdy[k] = invJ21 * dNdxi[k] + invJ22 * dNdeta[k];
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

                    for (int i = 0; i < 8; i++)
                        for (int j = 0; j < 8; j++) {
                            double val = 0.0;
                            for (int m = 0; m < 3; m++)
                                val += B[m][i] * DB[m][j];
                            Ke[i][j] += val * detJ * w;
                        }
                }
            }

            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 8; j++)
                    K[dofs[i] * num_dofs + dofs[j]] += Ke[i][j];
        }
    }

    double force_per_node = total_y / (ny + 1);
    for (int j = 0; j <= ny; j++) {
        int node = j * (nx + 1) + nx;
        F[2 * node + 1] += force_per_node;
    }

    for (int j = 0; j <= ny; j++) {
        int node = j * (nx + 1) + 0;
        int dof_x = 2 * node;
        int dof_y = 2 * node + 1;
        for (int k = 0; k < num_dofs; k++) {
            K[dof_x * num_dofs + k] = 0.0;
            K[k * num_dofs + dof_x] = 0.0;
            K[dof_y * num_dofs + k] = 0.0;
            K[k * num_dofs + dof_y] = 0.0;
        }
        K[dof_x * num_dofs + dof_x] = 1.0;
        K[dof_y * num_dofs + dof_y] = 1.0;
        F[dof_x] = 0.0;
        F[dof_y] = 0.0;
    }

    for (int col = 0; col < num_dofs; col++) {
        int max_row = col;
        double max_val = fabs(K[col * num_dofs + col]);
        for (int row = col + 1; row < num_dofs; row++) {
            double val = fabs(K[row * num_dofs + col]);
            if (val > max_val) {
                max_val = val;
                max_row = row;
            }
        }
        if (max_row != col) {
            for (int k = 0; k < num_dofs; k++) {
                double tmp = K[col * num_dofs + k];
                K[col * num_dofs + k] = K[max_row * num_dofs + k];
                K[max_row * num_dofs + k] = tmp;
            }
            double tmp = F[col];
            F[col] = F[max_row];
            F[max_row] = tmp;
        }
        double pivot = K[col * num_dofs + col];
        for (int row = col + 1; row < num_dofs; row++) {
            double fac = K[row * num_dofs + col] / pivot;
            for (int k = col; k < num_dofs; k++)
                K[row * num_dofs + k] -= fac * K[col * num_dofs + k];
            F[row] -= fac * F[col];
        }
    }

    double *u = (double *)calloc(num_dofs, sizeof(double));
    for (int i = num_dofs - 1; i >= 0; i--) {
        double sum = F[i];
        for (int j = i + 1; j < num_dofs; j++)
            sum -= K[i * num_dofs + j] * u[j];
        u[i] = sum / K[i * num_dofs + i];
    }

    double sum_uy = 0.0;
    for (int j = 0; j <= ny; j++) {
        int node = j * (nx + 1) + nx;
        sum_uy += u[2 * node + 1];
    }
    double avg_uy = sum_uy / (ny + 1);

    printf("tip_uy=%.10g\n", fabs(avg_uy));

    free(K);
    free(F);
    free(u);
    return 0;
}