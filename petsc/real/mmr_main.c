#include "globals.h"

PetscReal damp_func_real(PetscReal mu, PetscReal w)
{
    return mu/(mu*mu + w*w);
}

PetscReal damp_func_imag(PetscReal mu, PetscReal w)
{
    return -mu*w/(mu*mu + w*w);
}

int main(int argc, char **args)
{
    // Initialise
    SlepcInitialize(&argc, &args, NULL, NULL);

    // POD tolerance
    PetscInt pod_tolerance = 0;
    // Biot damping function realxation parameter
    //PetscReal mu = 4129.28;
    PetscReal mu = 1;

    // Filenames of matrices and vectors
    char mass_file[] = "./toy/matrices/mass.dat";
    char stiff_file[] = "./toy/matrices/stiffness.dat";
    char damp_file[] = "./toy/matrices/damping.dat";
    char load_file[] = "./toy/matrices/load.dat";
    char q_file[100];

    // Full matrices and vectors
    Mat M0, C0, K0;
    Mat M, C1, C2, K, Cw;
    Vec b0, b, *Qr, *Qi;

    // Reduced matrices and vectors
    Mat Mr, Cr, Kr;
    Vec br, *ur;

    // Frequency domain
    PetscReal omega_0, omega_i, omega_f, *omega;
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
    // Initialise interpolation points
    ind_ip_tmp = linspace(0, omega_len-1, n_ip+1);
    PetscMalloc1(n_ip, &ind_ip);
    for(i=0; i<n_ip; i++)
    {
        ind_ip[i] = (PetscInt)round((ind_ip_tmp[i]+ind_ip_tmp[i+1])/2);
    }

    // Read matrices from disk
    read_mat_file(MPI_COMM_WORLD, mass_file, &M0);
    read_mat_file(MPI_COMM_WORLD, stiff_file, &K0);
    read_mat_file(MPI_COMM_WORLD, damp_file, &C0);
    read_vec_file(MPI_COMM_WORLD, load_file, &b0);

    // Create block matrices
    create_block_mass(PETSC_COMM_WORLD, &M0, &M);
    create_block_stiffness(PETSC_COMM_WORLD, &K0, &K);
    create_block_damping(PETSC_COMM_WORLD, &C0, &C1, &C2);
    create_block_load(PETSC_COMM_WORLD, &b0, &b);

    // Destroy small matrices
    MatDestroy(&M0); MatDestroy(&C0);
    MatDestroy(&K0); VecDestroy(&b0);

    // Allocate eigenvectors (real and imaginary parts)
    PetscMalloc1(n_ip*n_eig, &Qr);
    PetscMalloc1(n_ip*n_eig, &Qi);

    n_tot=0;
    for(i=0; i<n_ip; i++)
    {
        omega_0 = omega[ind_ip[i]];

        PetscPrintf(PETSC_COMM_WORLD, "Eigenproblem at interpolation point #%d ...\n", i+1);
        MatDuplicate(C1, MAT_COPY_VALUES, &Cw);
        MatScale(Cw, damp_func_real(mu, omega_0));
        MatAXPY(Cw, damp_func_imag(mu, omega_0), C2, DIFFERENT_NONZERO_PATTERN);

        eigensolve(PETSC_COMM_WORLD, M, Cw, K, omega_0, n_eig, &(Qr[n_tot]), &(Qi[n_tot]), &nq);
        n_tot += nq;
    }

    // Finalise
    SlepcFinalize();
    return 0;
}

