#ifndef GLOBALS_H
#define GLOBALS_H

//#include <stdio.h>
//#include <stdlib.h>
//#include <math.h>

// petsc headers
#include <petscsys.h>
#include <petscmat.h>
#include <petscvec.h>
#include <petscksp.h>
#include <petscviewer.h>
#include <petscmath.h>

// slepc headers
#include <slepcpep.h>
#include <slepceps.h>

// File I/O functions - file_io.c
void read_mat_file(MPI_Comm comm, const char filename[], Mat *A);
void read_vec_file(MPI_Comm comm, const char filename[], Vec *b);
void write_mat_file(MPI_Comm comm, const char filename[], Mat *A);
void write_vec_file(MPI_Comm comm, const char filename[], Vec *b);

// POD functions for orthogonalisation
void get_covariance(MPI_Comm comm, Vec *Q, PetscInt n, Mat *R);
void get_pod_eigenvectors(MPI_Comm comm, Mat *A, PetscScalar tol,
        Vec **xr, PetscInt *rank);
void pod_orthogonalise(MPI_Comm comm, Vec *Q, PetscInt n_q, PetscScalar tol,
        Vec **Q1, PetscInt *rank);
void check_orthogonality(MPI_Comm comm, Vec *Q, PetscInt n_q);

// Reduce and solve - reduce.c
void direct_solve_dense(MPI_Comm comm, Mat *A, Vec *b, Vec *u);
void project_matrix(MPI_Comm comm, Mat *M, Vec *Q, PetscInt n_q, Mat *A);
void project_vector(MPI_Comm comm, Vec *u, Vec *Q, PetscInt n_q, Vec *u_new);

void direct_sweep(MPI_Comm comm, Mat *M, Mat *C1, Mat *C2, Mat *K, Vec *b,
        PetscScalar omega_i, PetscScalar omega_f, PetscInt n_omega,
        PetscScalar mu, Vec **u);

void recover_vector(MPI_Comm comm, Vec *u, Vec *Q, PetscInt n_q, Vec *u_new);
void recover_vectors(MPI_Comm comm, Vec *u, PetscInt n_u, Vec *Q, PetscInt n_q,
        Vec **u_new);

void check_projection(MPI_Comm comm, Vec u, Vec *Q, PetscInt n_q);

// Eigenvalue problem - eigen.c
void eigensolve(MPI_Comm comm, Mat M, Mat C, Mat K, PetscInt n_eig, Vec *Qr, Vec *Qi, PetscInt *nq);


// Miscellaneous - misc.c
PetscReal* linspace(PetscReal start, PetscReal stop, PetscInt len);

#endif
