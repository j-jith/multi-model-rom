#include "globals.h"

void eigensolve(MPI_Comm comm, Mat M, Mat C, Mat K, PetscReal ip, PetscInt n_eig, Vec *Qr, Vec *Qi, PetscInt *nq)
{
    Mat A[3];
    PEP pep;
    PEPType type;
    PetscScalar kr, ki;
    //Vec xr, xi;
    PetscInt i, its, nev, nconv;
    PetscReal err, re, im;

    EPS eps; ST st;
    KSP ksp; PC pc;

    // Set up problem
    PEPCreate(comm, &pep);
    A[0] = K; A[1] = C; A[2] = M;
    PEPSetOperators(pep, 3, A);

    // Problem type
    //PEPSetProblemType(pep, PEP_HERMITIAN);

    // Solver type
    PEPSetType(pep, PEPQARNOLDI);
    PEPGetST(pep, &st);
    STSetTransform(st, PETSC_TRUE);
    //STSetShift(st, 0.0);

    //PEPSetType(pep, PEPLINEAR);
    //PEPLinearSetCompanionForm(pep, 1);
    //PEPLinearSetExplicitMatrix(pep, PETSC_TRUE);

    /*
    // Linear
    PEPSetType(pep, PEPLINEAR);
    PEPLinearGetEPS(pep, &eps);
    EPSSetType(eps, EPSKRYLOVSCHUR);
    EPSGetST(eps, &st);
    STSetType(st, STSINVERT);
    //EPSSetWhichEigenpairs(eps, EPS_TARGET_MAGNITUDE);
    //EPSSetTarget(eps, ip*ip);
    EPSSetWhichEigenpairs(eps, EPS_SMALLEST_IMAGINARY);

    //PEPSetType(pep, PEPQARNOLDI);
    //ST st; PEPGetST(pep, &st);
    //STSetTransform(st, PETSC_TRUE);
    //STSetShift(st, 0.0);

    // Direct factorisation with MUMPS
    STGetKSP(st, &ksp);
    KSPSetType(ksp, KSPPREONLY);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCLU);
    PCFactorSetMatSolverPackage(pc, MATSOLVERMUMPS);
    */

    // Which eigenvalues to compute
    PEPSetWhichEigenpairs(pep, PEP_SMALLEST_MAGNITUDE);
    //PEPSetWhichEigenpairs(pep, PEP_TARGET_MAGNITUDE);
    //PEPSetTarget(pep, 0.0);

    // No. of eigenvalues to compute
    PEPSetDimensions(pep, n_eig, PETSC_DEFAULT, PETSC_DEFAULT);
    // Tolerance and max. iterations
    //PEPSetTolerances(pep, 1e-10, 1000);

    // Solve eigenproblem
    PetscPrintf(comm, "Solving...");
    PEPSolve(pep);
    PetscPrintf(comm, " Done\n");

    // Show info about solution
    PEPGetType(pep, &type);
    PetscPrintf(comm, "Solution method: %s\n", type);
    PEPGetIterationNumber(pep, &its);
    PetscPrintf(comm, "Number of iterations of the method: %D\n", its);
    PEPGetDimensions(pep, &nev, NULL, NULL);
    PetscPrintf(comm, "Number of requested eigenvalues: %D\n", nev);

    // Get no. of converged eigenvalues
    PEPGetConverged(pep, &nconv);
    PetscPrintf(comm, "Number of converged eigenvalues: %D\n", nconv);

    // Only return the requested no. of eigenvectors
    if (nconv > nev)
        nconv = nev;

    // No. of eigenvalues returned
    *nq = nconv;

    if (nconv > 0)
    {
        PetscPrintf(comm,
                "\n           k          ||P(k)x||/||kx||\n"
                "   ----------------- ------------------\n");
        for(i=0; i<nconv; i++)
        {
            MatCreateVecs(A[0], &(Qr[i]), &(Qi[i]));
            PEPGetEigenpair(pep, i, &kr, &ki, Qr[i], Qi[i]);
            PEPComputeError(pep, i, PEP_ERROR_RELATIVE, &err);
#if defined(PETSC_USE_COMPLEX)
            re = PetscRealPart(kr);
            im = PetscImaginaryPart(kr);
#else
            re = kr;
            im = ki;
#endif
            if(im != 0.0)
            {
                PetscPrintf(comm,
                        " %9f%+9fi   %12g\n", (double)re, (double)im, (double)err);
            }
            else
            {
                PetscPrintf(comm,
                        "   %12f       %12g\n", (double)re, (double)err);
            }
        }
        PetscPrintf(comm, "\n");
    }
    else
    {
        PetscPrintf(comm, "Convergence failure\n");
    }

    PEPDestroy(&pep);
}
