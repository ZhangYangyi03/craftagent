#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv){
    if(argc < 6) return 1;

    double L       = atof(argv[1]);
    double H       = atof(argv[2]);
    int    nx      = atoi(argv[3]);
    int    ny      = atoi(argv[4]);
    double total_y = atof(argv[5]);

    if(nx < 1 || ny < 1) return 1;

    int npx   = nx + 1;
    int npy   = ny + 1;
    int nnode = npx * npy;
    int ndof  = 2 * nnode;

    /* 节点坐标，行优先编号 */
    double *X = (double*)malloc((size_t)nnode * sizeof(double));
    double *Y = (double*)malloc((size_t)nnode * sizeof(double));
    int i, j;
    for(j = 0; j < npy; j++){
        for(i = 0; i < npx; i++){
            int id = j * npx + i;
            X[id] = L * (double)i / (double)nx;
            Y[id] = H * (double)j / (double)ny;
        }
    }

    /* 平面应变材料矩阵 */
    const double E  = 210000.0;
    const double nu = 0.3;
    const double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const double mu     = E / (2.0 * (1.0 + nu));

    double D[3][3];
    D[0][0] = lambda + 2.0 * mu; D[0][1] = lambda;        D[0][2] = 0.0;
    D[1][0] = lambda;            D[1][1] = lambda + 2.0 * mu; D[1][2] = 0.0;
    D[2][0] = 0.0;               D[2][1] = 0.0;           D[2][2] = mu;

    double *K = (double*)calloc((size_t)ndof * (size_t)ndof, sizeof(double));
    double *F = (double*)calloc((size_t)ndof, sizeof(double));

    /* 2x2 高斯积分 */
    const double g = 1.0 / sqrt(3.0);
    double gpts[2] = { -g, g };
    double gwt[2]  = { 1.0, 1.0 };

    /* Q4 节点自然坐标符号 */
    int xi_s[4] = { -1,  1,  1, -1 };
    int et_s[4] = { -1, -1,  1,  1 };

    int ex, ey;
    for(ey = 0; ey < ny; ey++){
        for(ex = 0; ex < nx; ex++){
            int nn[4];
            nn[0] =  ey      * npx + ex;
            nn[1] =  ey      * npx + ex + 1;
            nn[2] = (ey + 1) * npx + ex + 1;
            nn[3] = (ey + 1) * npx + ex;

            double Ke[8][8];
            int r, c;
            for(r = 0; r < 8; r++)
                for(c = 0; c < 8; c++) Ke[r][c] = 0.0;

            int a, b;
            for(a = 0; a < 2; a++){
                for(b = 0; b < 2; b++){
                    double xi  = gpts[a];
                    double eta = gpts[b];
                    double wq  = gwt[a] * gwt[b];

                    double dNxi[4], dNet[4];
                    int k;
                    for(k = 0; k < 4; k++){
                        dNxi[k] = 0.25 * xi_s[k] * (1.0 + et_s[k] * eta);
                        dNet[k] = 0.25 * et_s[k] * (1.0 + xi_s[k] * xi);
                    }

                    double J00 = 0.0, J01 = 0.0, J10 = 0.0, J11 = 0.0;
                    for(k = 0; k < 4; k++){
                        int id = nn[k];
                        J00 += dNxi[k] * X[id];
                        J01 += dNxi[k] * Y[id];
                        J10 += dNet[k] * X[id];
                        J11 += dNet[k] * Y[id];
                    }
                    double detJ = J00 * J11 - J01 * J10;

                    double iJ00 =  J11 / detJ;
                    double iJ01 = -J01 / detJ;
                    double iJ10 = -J10 / detJ;
                    double iJ11 =  J00 / detJ;

                    double dNdx[4], dNdy[4];
                    for(k = 0; k < 4; k++){
                        dNdx[k] = iJ00 * dNxi[k] + iJ10 * dNet[k];
                        dNdy[k] = iJ01 * dNxi[k] + iJ11 * dNet[k];
                    }

                    double B[3][8];
                    for(k = 0; k < 4; k++){
                        B[0][2*k]     = dNdx[k];
                        B[0][2*k + 1] = 0.0;
                        B[1][2*k]     = 0.0;
                        B[1][2*k + 1] = dNdy[k];
                        B[2][2*k]     = dNdy[k];
                        B[2][2*k + 1] = dNdx[k];
                    }

                    double DB[3][8];
                    int p, q;
                    for(p = 0; p < 3; p++){
                        for(r = 0; r < 8; r++){
                            double s = 0.0;
                            for(q = 0; q < 3; q++) s += D[p][q] * B[q][r];
                            DB[p][r] = s;
                        }
                    }

                    double scale = wq * detJ;
                    for(r = 0; r < 8; r++){
                        for(c = 0; c < 8; c++){
                            double s = 0.0;
                            for(p = 0; p < 3; p++) s += B[p][r] * DB[p][c];
                            Ke[r][c] += s * scale;
                        }
                    }
                }
            }

            int I, Jc;
            for(I = 0; I < 4; I++){
                for(Jc = 0; Jc < 4; Jc++){
                    int base_r = 2 * nn[I];
                    int base_c = 2 * nn[Jc];
                    int aa, bb;
                    for(aa = 0; aa < 2; aa++){
                        for(bb = 0; bb < 2; bb++){
                            K[(size_t)(base_r + aa) * ndof + (base_c + bb)]
                                += Ke[2*I + aa][2*Jc + bb];
                        }
                    }
                }
            }
        }
    }

    /* 载荷：x = L 边节点均分 total_y，作用于 uy 自由度 */
    double fy_node = total_y / (double)npy;
    for(j = 0; j < npy; j++){
        int id = j * npx + (npx - 1);
        F[2*id + 1] += fy_node;
    }

    /* Dirichlet：x = 0 边所有节点固定 */
    for(j = 0; j < npy; j++){
        int id = j * npx;
        int d1 = 2 * id;
        int d2 = 2 * id + 1;
        int r, c;
        for(r = 0; r < ndof; r++) K[(size_t)d1 * ndof + r] = 0.0;
        for(r = 0; r < ndof; r++) K[(size_t)d2 * ndof + r] = 0.0;
        for(c = 0; c < ndof; c++) K[(size_t)c * ndof + d1] = 0.0;
        for(c = 0; c < ndof; c++) K[(size_t)c * ndof + d2] = 0.0;
        K[(size_t)d1 * ndof + d1] = 1.0;
        K[(size_t)d2 * ndof + d2] = 1.0;
        F[d1] = 0.0;
        F[d2] = 0.0;
    }

    /* 带部分主元的高斯消元 */
    int col;
    for(col = 0; col < ndof; col++){
        int piv = col;
        double maxv = fabs(K[(size_t)col * ndof + col]);
        int r;
        for(r = col + 1; r < ndof; r++){
            double v = fabs(K[(size_t)r * ndof + col]);
            if(v > maxv){ maxv = v; piv = r; }
        }
        if(piv != col){
            int c;
            for(c = 0; c < ndof; c++){
                double tmp = K[(size_t)col * ndof + c];
                K[(size_t)col * ndof + c] = K[(size_t)piv * ndof + c];
                K[(size_t)piv * ndof + c] = tmp;
            }
            double tmp = F[col]; F[col] = F[piv]; F[piv] = tmp;
        }
        double pivval = K[(size_t)col * ndof + col];
        for(r = col + 1; r < ndof; r++){
            double factor = K[(size_t)r * ndof + col] / pivval;
            if(factor == 0.0) continue;
            int c;
            for(c = col; c < ndof; c++){
                K[(size_t)r * ndof + c] -= factor * K[(size_t)col * ndof + c];
            }
            F[r] -= factor * F[col];
        }
    }

    /* 回代 */
    double *u = (double*)calloc((size_t)ndof, sizeof(double));
    int r;
    for(r = ndof - 1; r >= 0; r--){
        double s = F[r];
        int c;
        for(c = r + 1; c < ndof; c++) s -= K[(size_t)r * ndof + c] * u[c];
        u[r] = s / K[(size_t)r * ndof + r];
    }

    /* x = L 边所有节点 uy 平均值的绝对值 */
    double sum_uy = 0.0;
    int cnt = 0;
    for(j = 0; j < npy; j++){
        int id = j * npx + (npx - 1);
        sum_uy += u[2*id + 1];
        cnt++;
    }
    double avg = sum_uy / (double)cnt;
    printf("tip_uy=%.10g\n", fabs(avg));

    free(X); free(Y); free(K); free(F); free(u);
    return 0;
}