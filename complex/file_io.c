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

int Mat_Parallel_Load(MPI_Comm comm,const char *name,Mat *newmat)
// http://www.mcs.anl.gov/petsc/petsc-current/src/mat/examples/tutorials/ex5.c
{
  Mat            A;
  PetscScalar    *vals;
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       i,j,rstart,rend;
  PetscInt       header[4],M,N,m;
  PetscInt       *ourlens,*offlens,jj,*mycols,maxnz;
  PetscInt       cend,cstart,n,*rowners;
  int            fd1,fd2;
  PetscViewer    viewer1,viewer2;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  /* Open the files; each process opens its own file */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,name,FILE_MODE_READ,&viewer1);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetDescriptor(viewer1,&fd1);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd1,(char*)header,4,PETSC_INT);CHKERRQ(ierr);

  /* open the file twice so that later we can read entries from two different parts of the
     file at the same time. Note that due to file caching this should not impact performance */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,name,FILE_MODE_READ,&viewer2);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetDescriptor(viewer2,&fd2);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd2,(char*)header,4,PETSC_INT);CHKERRQ(ierr);

  /* error checking on files */
  if (header[0] != MAT_FILE_CLASSID) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"not matrix object");
  ierr = MPI_Allreduce(header+2,&N,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
  if (N != size*header[2]) SETERRQ(PETSC_COMM_SELF,1,"All files must have matrices with the same number of total columns");

  /* number of rows in matrix is sum of rows in all files */
  m    = header[1]; N = header[2];
  ierr = MPI_Allreduce(&m,&M,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);

  /* determine rows of matrices owned by each process */
  ierr       = PetscMalloc1(size+1,&rowners);CHKERRQ(ierr);
  ierr       = MPI_Allgather(&m,1,MPIU_INT,rowners+1,1,MPIU_INT,comm);CHKERRQ(ierr);
  rowners[0] = 0;
  for (i=2; i<=size; i++) {
    rowners[i] += rowners[i-1];
  }
  rstart = rowners[rank];
  rend   = rowners[rank+1];
  ierr   = PetscFree(rowners);CHKERRQ(ierr);

  /* determine column ownership if matrix is not square */
  if (N != M) {
    n      = N/size + ((N % size) > rank);
    ierr   = MPI_Scan(&n,&cend,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
    cstart = cend - n;
  } else {
    cstart = rstart;
    cend   = rend;
    n      = cend - cstart;
  }

  /* read in local row lengths */
  ierr = PetscMalloc1(m,&ourlens);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&offlens);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd1,ourlens,m,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd2,ourlens,m,PETSC_INT);CHKERRQ(ierr);

  /* determine buffer space needed for column indices of any one row*/
  maxnz = 0;
  for (i=0; i<m; i++) {
    maxnz = PetscMax(maxnz,ourlens[i]);
  }

  /* allocate enough memory to hold a single row of column indices */
  ierr = PetscMalloc1(maxnz,&mycols);CHKERRQ(ierr);

  /* loop over local rows, determining number of off diagonal entries */
  ierr = PetscMemzero(offlens,m*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    ierr = PetscBinaryRead(fd1,mycols,ourlens[i],PETSC_INT);CHKERRQ(ierr);
    for (j=0; j<ourlens[i]; j++) {
      if (mycols[j] < cstart || mycols[j] >= cend) offlens[i]++;
    }
  }

  /* on diagonal entries are all that were not counted as off-diagonal */
  for (i=0; i<m; i++) {
    ourlens[i] -= offlens[i];
  }

  /* create our matrix */
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(A,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,0,ourlens,0,offlens);CHKERRQ(ierr);

  for (i=0; i<m; i++) {
    ourlens[i] += offlens[i];
  }
  ierr = PetscFree(offlens);CHKERRQ(ierr);

  /* allocate enough memory to hold a single row of matrix values */
  ierr = PetscMalloc1(maxnz,&vals);CHKERRQ(ierr);

  /* read in my part of the matrix numerical values and columns 1 row at a time and put in matrix  */
  jj = rstart;
  for (i=0; i<m; i++) {
    //ierr = PetscBinaryRead(fd1,vals,ourlens[i],PETSC_SCALAR);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd1,vals,ourlens[i],PETSC_DOUBLE);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd2,mycols,ourlens[i],PETSC_INT);CHKERRQ(ierr);
    ierr = MatSetValues(A,1,&jj,ourlens[i],mycols,vals,INSERT_VALUES);CHKERRQ(ierr);
    jj++;
  }
  ierr = PetscFree(ourlens);CHKERRQ(ierr);
  ierr = PetscFree(vals);CHKERRQ(ierr);
  ierr = PetscFree(mycols);CHKERRQ(ierr);

  ierr    = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr    = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *newmat = A;
  ierr    = PetscViewerDestroy(&viewer1);CHKERRQ(ierr);
  ierr    = PetscViewerDestroy(&viewer2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
