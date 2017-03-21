#include <petscsys.h>
#include <petscmat.h>
#include <petscvec.h>
#include <petscksp.h>
#include <petscviewer.h>
#include <petscmath.h>

PetscReal* gen_random_numbers(MPI_Comm comm, PetscInt N, PetscReal lower, PetscReal upper)
{
    PetscRandom rnd;
    PetscScalar val;
    PetscReal *array;
    PetscInt i;

    PetscRandomCreate(comm, &rnd);
    PetscRandomSetInterval(rnd, lower, upper);

    PetscMalloc1(N, &array);
    for(i=0; i<N; i++)
    {
        PetscRandomGetValue(rnd, &val);
        array[i] = PetscRealPart(val);
    }

    PetscRandomDestroy(&rnd);

    return array;
}

void write_mat_file(MPI_Comm comm, const char filename[], Mat *A)
{
    PetscInt n_rows, n_cols;
    PetscPrintf(comm, "Writing matrix to %s ...\n", filename);

    MatGetSize(*A, &n_rows, &n_cols);
    PetscPrintf(comm, "Shape: (%d, %d)\n", n_rows, n_cols);

    PetscViewer viewer;
    PetscViewerBinaryOpen(comm, filename, FILE_MODE_WRITE, &viewer);

    MatView(*A, viewer);

    PetscViewerDestroy(&viewer);
}

void write_vec_file(MPI_Comm comm, const char filename[], Vec *b)
{
    PetscInt n_rows = 0;
    PetscPrintf(comm, "Writing vector to %s ...\n", filename);

    VecGetSize(*b, &n_rows);
    PetscPrintf(comm, "Shape: (%d, )\n", n_rows);

    PetscViewer viewer;
    PetscViewerBinaryOpen(comm, filename, FILE_MODE_WRITE, &viewer);

    VecView(*b, viewer);

    PetscViewerDestroy(&viewer);
}

int main(int argc, char **args)
{
    /*======Parameters======*/
    // Degrees of freedom
    PetscInt N=1000;
    // Point load
    PetscReal load=1.0;
    // Rayleigh damping parameters
    PetscReal alpha=0.001, beta=0.01;
    /*======================*/

    // Initialise
    PetscInitialize(&argc, &args, NULL, NULL);
    MPI_Comm comm = MPI_COMM_WORLD;

    // mass, damping and stiffness matrices
    Mat M, C, K;
    // Load vector
    Vec f;

    // point masses and stiffnesses
    PetscReal *m, *k;

    //counters
    PetscInt i;

    // Random number generation
    m = gen_random_numbers(comm, N, 0.1, 1.0);
    k = gen_random_numbers(comm, N, 0.1, 1.0);

    // Preallocate memory for M and K
    MatCreateSeqAIJ(comm, N, N, 1, NULL, &M);
    MatCreateSeqAIJ(comm, N, N, 3, NULL, &K);

    // Assemble matrices
    for(i=0; i<N-1; i++)
    {
        MatSetValue(M, i, i, m[i], INSERT_VALUES);

        MatSetValue(K, i, i, k[i] + k[i+1], INSERT_VALUES);
        MatSetValue(K, i, i-1, -k[i], INSERT_VALUES);
        MatSetValue(K, i, i+1, -k[i+1], INSERT_VALUES);
    }
    MatSetValue(M, N-1, N-1, m[N-1], INSERT_VALUES);
    MatSetValue(K, N-1, N-1, k[N-1], INSERT_VALUES);
    MatSetValue(K, N-1, N-2, -k[N-1], INSERT_VALUES);
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);

    // Create damping matrix
    MatDuplicate(M, MAT_COPY_VALUES, &C);
    MatScale(C, alpha);
    MatAXPY(C, beta, K, DIFFERENT_NONZERO_PATTERN);

    // Create load vector
    VecCreateSeq(comm, N, &f);
    VecSetValue(f, N-1, load, INSERT_VALUES);
    VecAssemblyBegin(f);
    VecAssemblyEnd(f);

    // Write to file
    write_mat_file(comm, "matrices/mass.dat", &M);
    write_mat_file(comm, "matrices/damping.dat", &C);
    write_mat_file(comm, "matrices/stiffness.dat", &K);
    write_vec_file(comm, "matrices/load.dat", &f);

    // Clean up
    MatDestroy(&M);
    MatDestroy(&C);
    MatDestroy(&K);
    VecDestroy(&f);
    PetscFree(m);
    PetscFree(k);

    // Finalise
    PetscFinalize();
    return 0;
}
