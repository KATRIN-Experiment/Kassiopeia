
/* Program usage:  mpiexec ex1 [-help] [all PETSc options] */

static char help[] = "Solves a tridiagonal linear system with KSP.\n\n";

/*T
   Concepts: KSP^solving a system of linear equations
   Processors: 1
T*/

/* 
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners

  Note:  The corresponding parallel example is ex23.c
*/
// #include <petscksp.h>

// PetscErrorCode MatrixFreeMult(Mat,Vec,Vec);

int main(int argc, char** args)
{
    // Vec            x, b, u;      /* approx solution, RHS, exact solution */
    // Mat            A;            /* linear system matrix */
    // KSP            ksp;         /* linear solver context */
    // PC             pc;           /* preconditioner context */
    // PetscReal      norm,tol=1.e-14;  /* norm of solution error */
    // PetscErrorCode ierr;
    // PetscInt       i,n = 50,col[3],its;
    // PetscMPIInt    size;
    // PetscScalar    neg_one = -1.0,one = 1.0,value[3];
    // PetscBool      nonzeroguess = PETSC_FALSE;

    // /* Initialization for MPI + PETSc */
    // PetscInitialize(&argc,&args,(char *)0,help);

    // /* to determine the number of processes */
    // ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

    // /* if the number of processes is >1, abort */
    // if (size != 1) SETERRQ(PETSC_COMM_WORLD,1,"This is a uniprocessor example only!");

    // /* Set the size of the matrix */
    // ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);

    // ierr = PetscOptionsGetBool(PETSC_NULL,"-nonzero_guess",&nonzeroguess,PETSC_NULL);CHKERRQ(ierr);


    // /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //        Compute the matrix and right-hand-side vector that define
    //        the linear system, Ax = b.
    //    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    // /*
    //    Create vectors.  Note that we form 1 vector from scratch and
    //    then duplicate as needed.
    // */
    // ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
    // ierr = PetscObjectSetName((PetscObject) x, "Solution");CHKERRQ(ierr);
    // ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
    // ierr = VecSetFromOptions(x);CHKERRQ(ierr);
    // ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
    // ierr = VecDuplicate(x,&u);CHKERRQ(ierr);

    // /*
    //    Create matrix.  When using MatCreate(), the matrix format can
    //    be specified at runtime.

    //    Performance tuning note:  For problems of substantial size,
    //    preallocation of matrix memory is crucial for attaining good
    //    performance. See the matrix chapter of the users manual for details.
    // */
    // double matrix[n][n];
    // for (int i=0;i<n;i++)
    // {
    //   for (int j=0;j<n;j++)
    //   {
    //     if (j==i-1) matrix[i][j] = -1.;
    //     else if (j==i) matrix[i][j] = 2.;
    //     else if (j==i+1) matrix[i][j] = -1.;
    //   }
    // }

    // ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,NULL,&A);CHKERRQ(ierr);
    // ierr = MatShellSetContext(A,(void*)&matrix);
    // ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    // ierr = MatSetUp(A);CHKERRQ(ierr);

    // /*
    //    Assemble matrix
    // */
    // value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
    // for (i=1; i<n-1; i++) {
    //   col[0] = i-1; col[1] = i; col[2] = i+1;
    //   ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
    // }
    // i = n - 1; col[0] = n - 2; col[1] = n - 1;
    // ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    // i = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
    // ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    // ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    // ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    // /*
    //    Set exact solution; then compute right-hand-side vector.
    // */
    // ierr = VecSet(u,one);CHKERRQ(ierr);
    // ierr = MatMult(A,u,b);CHKERRQ(ierr);

    // /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //               Create the linear solver and set various options
    //    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    // /*
    //    Create linear solver context
    // */
    // ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);

    // /*
    //    Set operators. Here the matrix that defines the linear system
    //    also serves as the preconditioning matrix.
    // */
    // ierr = KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

    // /*
    //    Set linear solver defaults for this problem (optional).
    //    - By extracting the KSP and PC contexts from the KSP context,
    //      we can then directly call any KSP and PC routines to set
    //      various options.
    //    - The following four statements are optional; all of these
    //      parameters could alternatively be specified at runtime via
    //      KSPSetFromOptions();
    // */
    // ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    // ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
    // ierr = KSPSetTolerances(ksp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

    // /*
    //   Set runtime options, e.g.,
    //       -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    //   These options will override those specified above as long as
    //   KSPSetFromOptions() is called _after_ any other customization
    //   routines.
    // */
    // ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

    // if (nonzeroguess) {
    //   PetscScalar p = .5;
    //   ierr = VecSet(x,p);CHKERRQ(ierr);
    //   ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);
    // }

    // /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //                     Solve the linear system
    //    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    // /*
    //    Solve linear system
    // */
    // ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

    // /*
    //    View solver info; we could instead use the option -ksp_view to
    //    print this info to the screen at the conclusion of KSPSolve().
    // */
    // ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    // ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    // ierr = VecView(b,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    // ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    // /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //                     Check solution and clean up
    //    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    // /*
    //    Check the error
    // */
    // ierr = VecAXPY(x,neg_one,u);CHKERRQ(ierr);
    // ierr  = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
    // ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
    // if (norm > tol){
    //   ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %G, Iterations %D\n",
    //                    norm,its);CHKERRQ(ierr);
    // }

    // /*
    //    Free work space.  All PETSc objects should be destroyed when they
    //    are no longer needed.
    // */
    // ierr = VecDestroy(&x);CHKERRQ(ierr); ierr = VecDestroy(&u);CHKERRQ(ierr);
    // ierr = VecDestroy(&b);CHKERRQ(ierr); ierr = MatDestroy(&A);CHKERRQ(ierr);
    // ierr = KSPDestroy(&ksp);CHKERRQ(ierr);

    // /*
    //    Always call PetscFinalize() before exiting a program.  This routine
    //      - finalizes the PETSc libraries as well as MPI
    //      - provides summary and diagnostic information if certain runtime
    //        options are chosen (e.g., -log_summary).
    // */
    // ierr = PetscFinalize();
    return 0;
}

// PetscErrorCode MatrixFreeMult(Mat A,Vec x,Vec b)
// {
//   int i,j,ic,il,ista,iend;
//   double dx,dy,w;
//   PetscScalar *ax,*ay;
//   PetscErrorCode ierr;
//   BOTH *both;
//   ierr = MatShellGetContext(A, (void **) &both);CHKERRQ(ierr);
//   PARTICLE *particle = both->p;
//   CLUSTER *cluster = both->c;

//   PetscFunctionBegin;
//   ierr = VecGetArray(x,&ax);CHKERRQ(ierr);
//   ierr = VecGetArray(y,&ay);CHKERRQ(ierr);
//   for(i=particle->ista; i<particle->iend; i++) {
//     ierr = VecSetValues(particle->gi,1,&i,&ax[i-particle->ista],INSERT_VALUES);CHKERRQ(ierr);
//   }
//   ierr = VecAssemblyBegin(particle->gi);CHKERRQ(ierr);
//   ierr = VecAssemblyEnd(particle->gi);CHKERRQ(ierr);
//   ierr = VecGhostUpdateBegin(particle->gi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
//   ierr = VecGhostUpdateEnd(particle->gi,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
//   ierr = VecGetArray(particle->gi,&particle->gil);CHKERRQ(ierr);
//   for (ic=cluster->icsta; ic<cluster->icend; ic++) {
//     Get_trunc trunc;
//     trunc.get_trunc(particle,cluster,ic);
//     ista = cluster->ista[ic];
//     iend = cluster->iend[ic];
//     for (i=ista; i<=iend; i++) {
//       il = cluster->ilocal[i];
//       w = 0;
//       for (j=0; j<cluster->nptruncj; j++) {
//         dx = particle->xil[i]-cluster->xjt[j];
//         dy = particle->yil[i]-cluster->yjt[j];
//         w += cluster->gjt[j]*exp(-(dx*dx+dy*dy)/(2*particle->sigma*particle->sigma))/
//           (2*M_PI*particle->sigma*particle->sigma);
//       }
//       ay[il-particle->ista] = w;
//     }
//     /* Counted 1 for exp() */
//     ierr = PetscLogFlops((iend-ista)*cluster->nptruncj*15);CHKERRQ(ierr);
//   }
//   ierr = VecRestoreArray(particle->gi,&particle->gil);CHKERRQ(ierr);
//   ierr = VecRestoreArray(x,&ax);CHKERRQ(ierr);
//   ierr = VecRestoreArray(y,&ay);CHKERRQ(ierr);
//   PetscFunctionReturn(0);
// }
