#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static double *K;      /* dense global stiffness */
static double *F;
static double *U;
static int ndof;

static void solve(int n){
    /* Gaussian elimination with partial pivoting */
    int i,j,k,piv;
    double maxv,tmp;
    for(k=0;k<n;k++){
        piv=k; maxv=fabs(K[k*n+k]);
        for(i=k+1;i<n;i++){ double v=fabs(K[i*n+k]); if(v>maxv){maxv=v;piv=i;} }
        if(piv!=k){
            for(j=0;j<n;j++){ tmp=K[k*n+j]; K[k*n+j]=K[piv*n+j]; K[piv*n+j]=tmp; }
            tmp=F[k]; F[k]=F[piv]; F[piv]=tmp;
        }
        for(i=k+1;i<n;i++){
            double f=K[i*n+k]/K[k*n+k];
            if(f==0.0) continue;
            for(j=k;j<n;j++) K[i*n+j]-=f*K[k*n+j];
            F[i]-=f*F[k];
        }
    }
    for(i=n-1;i>=0;i--){
        double s=F[i];
        for(j=i+1;j<n;j++) s-=K[i*n+j]*U[j];
        U[i]=s/K[i*n+i];
    }
}

int main(int argc,char **argv){
    if(argc<6){ fprintf(stderr,"usage: %s L H nx ny total_y\n",argv[0]); return 1; }
    double L=atof(argv[1]);
    double H=atof(argv[2]);
    int nx=atoi(argv[3]);
    int ny=atoi(argv[4]);
    double total_y=atof(argv[5]);

    int npx=nx+1, npy=ny+1;
    int np=npx*npy;
    ndof=2*np;
    int e, i, j, a, b;

    /* coordinates */
    double *X=malloc(np*sizeof(double));
    double *Y=malloc(np*sizeof(double));
    for(j=0;j<npy;j++)
        for(i=0;i<npx;i++){
            int n=j*npx+i;
            X[n]=L*i/nx;
            Y[n]=H*j/ny;
        }

    /* material matrix (plane strain) */
    double E=210000.0, nu=0.3;
    double lam=E*nu/((1.0+nu)*(1.0-2.0*nu));
    double mu =E/(2.0*(1.0+nu));
    double D[3][3]={
        {lam+2.0*mu, lam,        0.0},
        {lam,        lam+2.0*mu, 0.0},
        {0.0,        0.0,        mu }
    };

    K=calloc((size_t)ndof*ndof,sizeof(double));
    F=calloc(ndof,sizeof(double));
    U=calloc(ndof,sizeof(double));

    /* load on x=L edge nodes */
    double fy=total_y/npy;
    for(j=0;j<npy;j++){
        int n=j*npx+nx;
        F[2*n+1]=fy;
    }

    /* element assembly, 2x2 Gauss */
    double gp=0.5773502691896257;
    double gs[4]={-gp,gp,-gp,gp};
    double gt[4]={-gp,-gp,gp,gp};
    double gw=1.0;
    double ke[8][8];
    int conn[4];

    for(e=0;e<nx*ny;e++){
        int ix=e%nx, iy=e/nx;
        conn[0]=iy*npx+ix;       /* (i,j)   */
        conn[1]=iy*npx+ix+1;     /* (i+1,j) */
        conn[2]=(iy+1)*npx+ix+1; /* (i+1,j+1) */
        conn[3]=(iy+1)*npx+ix;   /* (i,j+1) */
        double xe[4]={X[conn[0]],X[conn[1]],X[conn[2]],X[conn[3]]};
        double ye[4]={Y[conn[0]],Y[conn[1]],Y[conn[2]],Y[conn[3]]};

        for(i=0;i<8;i++) for(j=0;j<8;j++) ke[i][j]=0.0;

        for(a=0;a<4;a++){
            double s=gs[a], t=gt[a];
            /* shape function derivatives wrt xi, eta */
            double dNx[4],dNt[4];
            dNx[0]=-0.5*(1.0-t); dNx[1]= 0.5*(1.0-t);
            dNx[2]= 0.5*(1.0+t); dNx[3]=-0.5*(1.0+t);
            dNt[0]=-0.5*(1.0-s); dNt[1]=-0.5*(1.0+s);
            dNt[2]= 0.5*(1.0+s); dNt[3]= 0.5*(1.0-s);
            double Jxx=0,Jxt=0,Jtt=0,Jtx=0;
            for(i=0;i<4;i++){
                Jxx+=dNx[i]*xe[i]; Jxt+=dNx[i]*ye[i];
                Jtx+=dNt[i]*xe[i]; Jtt+=dNt[i]*ye[i];
            }
            double detJ=Jxx*Jtt-Jxt*Jtx;
            double dNxy[4][2];
            for(i=0;i<4;i++){
                dNxy[i][0]=( dNt[i]*Jxt - dNx[i]*Jtt)/detJ; /* dN/dx */
                dNxy[i][1]=( dNx[i]*Jtx - dNt[i]*Jxx)/detJ; /* dN/dy */
            }
            double w=gw*detJ;
            /* B matrix 3x8 */
            double B[3][8];
            for(i=0;i<4;i++){
                B[0][2*i]  =dNxy[i][0]; B[0][2*i+1]=0.0;
                B[1][2*i]  =0.0;        B[1][2*i+1]=dNxy[i][1];
                B[2][2*i]  =dNxy[i][1]; B[2][2*i+1]=dNxy[i][0];
            }
            for(i=0;i<8;i++)
                for(j=0;j<8;j++){
                    double s2=0.0;
                    for(b=0;b<3;b++) s2+=B[b][i]*D[b][0]*B[0][j]
                                        +B[b][i]*D[b][1]*B[1][j]
                                        +B[b][i]*D[b][2]*B[2][j];
                    ke[i][j]+=s2*w;
                }
        }
        /* scatter */
        for(i=0;i<4;i++)
            for(j=0;j<4;j++){
                int gi=2*conn[i], gj=2*conn[j];
                K[(gi  )*ndof+gj  ]+=ke[2*i  ][2*j  ];
                K[(gi  )*ndof+gj+1]+=ke[2*i  ][2*j+1];
                K[(gi+1)*ndof+gj  ]+=ke[2*i+1][2*j  ];
                K[(gi+1)*ndof+gj+1]+=ke[2*i+1][2*j+1];
            }
    }

    /* Dirichlet: x=0 nodes fixed */
    for(j=0;j<npy;j++){
        int n=j*npx;
        int dofs[2]={2*n,2*n+1};
        for(b=0;b<2;b++){
            int d=dofs[b];
            for(i=0;i<ndof;i++){ K[d*ndof+i]=0.0; K[i*ndof+d]=0.0; }
            K[d*ndof+d]=1.0;
            F[d]=0.0;
        }
    }

    solve(ndof);

    /* average uy on x=L */
    double sum=0.0;
    for(j=0;j<npy;j++) sum+=U[2*(j*npx+nx)+1];
    double avg=fabs(sum/npy);
    printf("tip_uy=%.10g\n",avg);

    free(X);free(Y);free(K);free(F);free(U);
    return 0;
}