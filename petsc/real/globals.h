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

// damping functions - mmr_main.c
PetscReal damp_func_real(PetscReal mu, PetscReal w);
PetscReal damp_func_imag(PetscReal mu, PetscReal w);

// File I/O functions - file_io.c
void read_mat_file(MPI_Comm comm, const char filename[], Mat *A);
void read_vec_file(MPI_Comm comm, const char filename[], Vec *b);
void write_mat_file(MPI_Comm comm, const char filename[], Mat *A);
void write_vec_file(MPI_Comm comm, const char filename[], Vec *b);

// Construct full (block) matrices from smaller matrices - block_matrices.c
void get_csr(MPI_Comm comm, Mat *M,
        PetscInt **ai, PetscInt **aj, PetscScalar **av,
        PetscInt *total_rows);

void get_block_diag_csr(MPI_Comm comm,
        PetscInt *ai, PetscInt *aj, PetscScalar *av, PetscInt nrows,
        PetscInt **ai1, PetscInt **aj1, PetscScalar **av1, PetscBool flip_signs);

void get_block_mass_csr(MPI_Comm comm,
        PetscInt *ai, PetscInt *aj, PetscScalar *av, PetscInt nrows,
        PetscInt **ai1, PetscInt **aj1, PetscScalar **av1);

void get_block_stiffness_csr(MPI_Comm comm,
        PetscInt *ai, PetscInt *aj, PetscScalar *av, PetscInt nrows,
        PetscInt **ai1, PetscInt **aj1, PetscScalar **av1);

void get_block_damping_csr(MPI_Comm comm,
        PetscInt *ai, PetscInt *aj, PetscScalar *av, PetscInt nrows,
        PetscInt **ai1, PetscInt **aj1, PetscScalar **av1,
        PetscInt **ai2, PetscInt **aj2, PetscScalar **av2);

void create_block_mass(MPI_Comm comm, Mat *M, Mat *M1);
void create_block_stiffness(MPI_Comm comm, Mat *M, Mat *M1);
void create_block_damping(MPI_Comm comm, Mat *M, Mat *M1, Mat *M2);
void create_block_load(MPI_Comm comm, Vec *f, Vec *f1);

// POD functions for orthogonalisation
void get_covariance(MPI_Comm comm, Vec *Q, PetscInt n, Mat *R);
void get_pod_eigenvectors(MPI_Comm comm, Mat *A, PetscReal tol,
        Vec **xr, PetscInt *rank);
void pod_orthogonalise(MPI_Comm comm, Vec *Q, PetscInt n_q, PetscReal tol,
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
void eigensolve(MPI_Comm comm, Mat M, Mat C, Mat K, PetscReal ip, PetscInt n_eig, Vec *Qr, Vec *Qi, PetscInt *nq);


// Miscellaneous - misc.c
PetscReal* linspace(PetscReal start, PetscReal stop, PetscInt len);

#endif
