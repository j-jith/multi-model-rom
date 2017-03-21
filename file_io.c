#include "globals.h"
//#include <yaml.h>

void read_mat_file(MPI_Comm comm, const char filename[], Mat *A)
{
    PetscInt n_rows = 0;
    PetscInt n_cols = 0;

    PetscPrintf(comm, "Reading matrix from %s ...\n", filename);

    PetscViewer viewer;
    PetscViewerBinaryOpen(comm, filename, FILE_MODE_READ, &viewer);

    MatCreate(comm, A);
    MatSetFromOptions(*A);
    MatLoad(*A, viewer);

    PetscViewerDestroy(&viewer);

    MatGetSize(*A, &n_rows, &n_cols);
    PetscPrintf(comm, "Shape: (%d, %d)\n", n_rows, n_cols);
}

void read_vec_file(MPI_Comm comm, const char filename[], Vec *b)
{
    PetscInt n_rows = 0;

    PetscPrintf(comm, "Reading vector from %s ...\n", filename);

    PetscViewer viewer;
    PetscViewerBinaryOpen(comm, filename, FILE_MODE_READ, &viewer);

    VecCreate(comm, b);
    VecSetFromOptions(*b);
    VecLoad(*b, viewer);

    PetscViewerDestroy(&viewer);

    VecGetSize(*b, &n_rows);
    PetscPrintf(comm, "Shape: (%d, )\n", n_rows);
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

