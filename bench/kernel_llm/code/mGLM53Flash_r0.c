```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Q4 shape functions on the biunit square and their natural derivatives. */
static void q4_shape(double xi, double eta, double N[4], double dN[4][2])
{
    N[0] = 0.25 * (1.0 - xi) * (1.0 - eta);
    N[1] = 0.25 * (1.0 + xi) * (1.0 - eta);
    N[2] = 0.25 * (1.0 + xi) * (1.0 + eta);
    N[3] = 0.25 * (1.0 - xi) * (1.0 + eta);

    /* dN[a][0] = dN_a/dxi , dN[a][1] = dN_a/deta */
    dN[0][0] = -0.25 * (1.0 - eta); dN[0][1] = -0.25 * (1.0 - xi);
    dN[1][0] =  0.25 * (1.0 - eta); dN[1][1] = -0.25 * (1.0 + xi);
    dN[2][0] =  0.25 * (1.0 + eta); dN[2][1] =  0.25 * (1.0 + xi);
    dN[3][0] = -0.25 * (1.0 + eta); dN[3][1] =  0.25 * (1.0 - xi);
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr, "usage: %s L H nx ny total_y\n", argv[0]);
        return EXIT_FAILURE;
    }

    double L       = strtod(argv[1], NULL);
    double H       = strtod(argv[2], NULL);
    long   nx      = strtol(argv[3], NULL, 10);
    long   ny      = strtol(argv[4], NULL, 10);
    double total_y = strtod(argv[5], NULL);

    if (!(L > 0.0) || !(H > 0.0) || nx < 1 || ny < 1) {
        fprintf(stderr, "invalid arguments\n");
        return EXIT_FAILURE;
    }

    const double E  = 210000.0;
    const double nu = 0.3;

    /* plane strain, isotropic */
    const double lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const double mu  = E / (2.0 * (1.0 + nu));
    const double D[3][3] = {
        { lam + 2.0 * mu, lam,            0.0 },
        { lam,            lam + 2.0 * mu, 0.0 },