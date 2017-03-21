#include "globals.h"

PetscComplex damp_func(PetscReal mu, PetscComplex s)
{
    return mu/(mu + s);
}

int main(int argc, char **args)
{
    // Initialise
    SlepcInitialize(&argc, &args, NULL, NULL);

    // POD tolerance
    PetscInt pod_tolerance = 0;
    // Biot damping function realxation parameter
    PetscReal mu = 4129.28;

    // Filenames of matrices and vectors
    char mass_file[] = "./matrices/mass.dat";
    char stiff_file[] = "./matrices/stiffness.dat";
    char damp_file[] = "./matrices/damping.dat";
    char load_file[] = "./matrices/force.dat";
    char q_file[100];

    // Full matrices and vectors
    Mat M, C, K, Cs;
    Vec b, *Qr, *Qi;

    // Reduced matrices and vectors
    Mat Mr, Cr, Kr;
    Vec br, *ur;

    // Frequency domain
    PetscReal omega_i, omega_f, *omega;
    PetscComplex *s;
    PetscInt omega_len;
    // Interpolation points
    PetscInt *ind_ip;
    PetscInt n_ip, n_eig;
    PetscReal *ind_ip_tmp;

    // Counters
    PetscInt i, j, nq, n_tot;

    // Parse arguments
    if(argc > 5)
    {
        omega_i = (PetscReal)atof(args[1])*2*M_PI;
        omega_f = (PetscReal)atof(args[2])*2*M_PI;
        omega_len = (PetscInt)atoi(args[3]);
        n_ip = (PetscInt)atoi(args[4]);
        n_eig = (PetscInt)atoi(args[5]);
    }
    else
    {
        //ind_ip = (PetscInt)round(omega_len/2);
        PetscPrintf(MPI_COMM_WORLD, "Usage: ./mmr <initial freq.> <final freq.> <no. of freqs.> <no. of interpolation points> <no. of eigenvectors per interpolation point>\n");
        return 0;
    }

    // Initialise freqeuncy domain
    omega = linspace(omega_i, omega_f, omega_len);
    PetscMalloc1(omega_len, &s);
    for(i=0; i<omega_len; i++)
    {
        s[i] = PETSC_i * omega[i];
    }
    // Initialise interpolation points
    ind_ip_tmp = linspace(0, omega_len-1, n_ip+1);
    PetscMalloc1(n_ip, &ind_ip);
    for(i=0; i<n_ip; i++)
    {
        ind_ip[i] = (PetscInt)round((ind_ip_tmp[i]+ind_ip_tmp[i+1])/2);
    }

    // Read matrices from disk
    //read_mat_file(MPI_COMM_WORLD, mass_file, &M);
    //read_mat_file(MPI_COMM_WORLD, stiff_file, &K);
    //read_mat_file(MPI_COMM_WORLD, damp_file, &C);
    //read_vec_file(MPI_COMM_WORLD, load_file, &b);
    Mat_Parallel_Load(MPI_COMM_WORLD, mass_file, &M);
    Mat_Parallel_Load(MPI_COMM_WORLD, stiff_file, &K);
    Mat_Parallel_Load(MPI_COMM_WORLD, damp_file, &C);
    //Mat_Parallel_Load(MPI_COMM_WORLD, load_file, &b);

    // Allocate eigenvectors (real and imaginary parts)
    PetscMalloc1(n_ip*n_eig, &Qr);
    PetscMalloc1(n_ip*n_eig, &Qi);

    n_tot=0;
    for(i=0; i<n_ip; i++)
    {
        PetscPrintf(PETSC_COMM_WORLD, "Eigenproblem at interpolation point #%d ...\n", i+1);
        MatDuplicate(C, MAT_COPY_VALUES, &Cs);
        MatScale(Cs, damp_func(mu, s[ind_ip[i]]));

        eigensolve(PETSC_COMM_WORLD, M, C, K, s[ind_ip[i]], n_eig, &(Qr[n_tot]), &(Qi[n_tot]), &nq);
        n_tot += nq;
    }

    // Finalise
    SlepcFinalize();
    return 0;
}

