#ifndef KPETSCSOLVER_DEF
#define KPETSCSOLVER_DEF

#include "KIterativeSolver.hh"
#include "KSquareMatrix.hh"
#include "KVector.hh"

#include <cassert>
#include <petscksp.h>

namespace KEMField
{
template<typename ValueType> class KPETScSolver : public KIterativeSolver<ValueType>
{
  public:
    typedef KSquareMatrix<ValueType> Matrix;
    typedef KVector<ValueType> Vector;

    KPETScSolver();
    virtual ~KPETScSolver() {}

    unsigned int Dimension() const
    {
        return fDimension;
    }

    void Solve(const Matrix& A, Vector& x, const Vector& b) const;

    void CacheMatrixElements(bool choice)
    {
        fCacheMatrixElements = choice;
    }

  protected:
    mutable unsigned int fDimension;

    bool fCacheMatrixElements;
};

PetscErrorCode KPETScMatrixMultiply(Mat A_, Vec x_, Vec y_)
{
    PetscErrorCode ierr;
    PetscInt rstart, rend, n;
    unsigned int i_local;

    const KSquareMatrix<PetscScalar>* A;
    MatShellGetContext(A_, &A);
    n = A->Dimension();

    VecScatter ctx;
    Vec x_gathered;
    VecScatterCreateToAll(x_, &ctx, &x_gathered);
    VecScatterBegin(ctx, x_, x_gathered, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(ctx, x_, x_gathered, INSERT_VALUES, SCATTER_FORWARD);

    ierr = MatGetOwnershipRange(A_, &rstart, &rend);

    const PetscScalar* x;
    PetscScalar* y;
    VecGetArrayRead(x_gathered, &x);
    VecGetArray(y_, &y);

    // Computes vector b in the equation A*x = b
    for (int i = rstart; i < rend; i++) {
        i_local = i - rstart;
        y[i_local] = 0.;
        for (int j = 0; j < n; j++)
            y[i_local] += A->operator()(i, j) * x[j];
    }

    ierr = VecRestoreArray(y_, &y);

    return ierr;
}

PetscErrorCode KPETScMatrixMultiplyTranspose(Mat A_, Vec x_, Vec y_)
{
    PetscErrorCode ierr;
    PetscInt rstart, rend, n;
    unsigned int i_local;

    const KSquareMatrix<PetscScalar>* A;
    MatShellGetContext(A_, &A);
    n = A->Dimension();

    VecScatter ctx;
    Vec x_gathered;
    VecScatterCreateToAll(x_, &ctx, &x_gathered);
    VecScatterBegin(ctx, x_, x_gathered, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(ctx, x_, x_gathered, INSERT_VALUES, SCATTER_FORWARD);

    ierr = MatGetOwnershipRange(A_, &rstart, &rend);

    const PetscScalar* x;
    PetscScalar* y;
    VecGetArrayRead(x_gathered, &x);
    VecGetArray(y_, &y);

    // Computes vector b in the equation A*x = b
    for (int i = rstart; i < rend; i++) {
        i_local = i - rstart;
        y[i_local] = 0.;
        for (int j = 0; j < n; j++)
            y[i_local] += A->operator()(j, i) * x[j];
    }

    ierr = VecRestoreArray(y_, &y);

    return ierr;
}

PetscErrorCode PETScPreconditioner(PC, Vec x, Vec y)
{
    VecCopy(x, y);
    return 0;
}

template<typename ValueType> KPETScSolver<ValueType>::KPETScSolver() : fDimension(0), fCacheMatrixElements(false)
{
    assert(sizeof(PetscScalar) == sizeof(ValueType));
}

template<typename ValueType> void KPETScSolver<ValueType>::Solve(const Matrix& A, Vector& x, const Vector& b) const
{
    fDimension = b.Dimension();

    Vec x_, b_; /* approx solution, RHS */
    Mat A_;     /* linear system matrix */
    KSP ksp;    /* linear solver context */
    PC pc;      /* preconditioner context */
    PetscErrorCode ierr;
    PetscInt n = A.Dimension(), rstart, rend, nlocal;
    PetscScalar element;


    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Compute the matrix and right-hand-side vector that define
       the linear system, Ax = b.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
      Create vectors.  Note that we form 1 vector from scratch and
      then duplicate as needed. For this simple case let PETSc decide how
      many elements of the vector are stored on each processor. The second
      argument to VecSetSizes() below causes PETSc to decide.
    */
    ierr = VecCreate(PETSC_COMM_WORLD, &x_);
    ierr = VecSetSizes(x_, PETSC_DECIDE, n);
    ierr = VecSetFromOptions(x_);
    ierr = VecDuplicate(x_, &b_);

    /* Identify the starting and ending mesh points on each
       processor for the interior part of the mesh. We let PETSc decide
       above. */

    ierr = VecGetOwnershipRange(x_, &rstart, &rend);
    ierr = VecGetLocalSize(x_, &nlocal);

    if (!fCacheMatrixElements) {
        /*
	Create matrix.  When using MatCreate(), the matrix format can
	be specified at runtime.

	Performance tuning note:  For problems of substantial size,
	preallocation of matrix memory is crucial for attaining good
	performance. See the matrix chapter of the users manual for details.

	We pass in nlocal as the "local" size of the matrix to force it
	to have the same parallel layout as the vector created above.
      */
        ierr = MatCreate(PETSC_COMM_WORLD, &A_);
        ierr = MatSetSizes(A_, nlocal, nlocal, n, n);
        ierr = MatSetFromOptions(A_);
        ierr = MatSetUp(A_);

        /*
	Assemble matrix.

	The linear system is distributed across the processors by
	chunks of contiguous rows, which correspond to contiguous
	sections of the mesh on which the problem is discretized.
	For matrix assembly, each processor contributes entries for
	the part that it owns locally.
      */

        for (int i = rstart; i < rend; i++) {
            for (int j = 0; j < n; j++) {
                element = A(i, j);
                ierr = MatSetValues(A_, 1, &i, 1, &j, &element, INSERT_VALUES);
            }
        }

        /* Assemble the matrix */
        ierr = MatAssemblyBegin(A_, MAT_FINAL_ASSEMBLY);
        ierr = MatAssemblyEnd(A_, MAT_FINAL_ASSEMBLY);
    }

    /*
      Set initial solution and right-hand-side vector.
    */

    for (int j = 0; j < n; j++) {
        element = b(j);
        ierr = VecSetValues(b_, 1, &j, &element, INSERT_VALUES);
        element = x(j);
        ierr = VecSetValues(x_, 1, &j, &element, INSERT_VALUES);
    }

    VecAssemblyBegin(b_);
    VecAssemblyEnd(b_);

    VecAssemblyBegin(x_);
    VecAssemblyEnd(x_);

    if (fCacheMatrixElements) {
        PetscInt local_m, local_n;
        VecGetLocalSize(b_, &local_m);
        VecGetLocalSize(x_, &local_n);

        void* matrixData = const_cast<Matrix*>(&A);

        MatCreateShell(PETSC_COMM_WORLD, local_m, local_n, n, n, matrixData, &A_);
        MatShellSetOperation(A_, MATOP_MULT, (void (*)(void)) KPETScMatrixMultiply);
        MatShellSetOperation(A_, MATOP_MULT_TRANSPOSE, (void (*)(void)) KPETScMatrixMultiplyTranspose);
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create the linear solver and set various options
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
      Create linear solver context
    */
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);
    //ierr = KSPSetType(ksp,KSPBICG);
    ierr = KSPSetType(ksp, KSPGMRES);

    /*
      Set operators. Here the matrix that defines the linear system
      also serves as the preconditioning matrix.
    */
    ierr = KSPSetOperators(ksp, A_, A_, DIFFERENT_NONZERO_PATTERN);

    /*
      Set linear solver defaults for this problem (optional).
      - By extracting the KSP and PC contexts from the KSP context,
      we can then directly call any KSP and PC routines to set
      various options.
      - The following four statements are optional; all of these
      parameters could alternatively be specified at runtime via
      KSPSetFromOptions();
    */
    ierr = KSPGetPC(ksp, &pc);
    ierr = PCSetType(pc, PCNONE);


    //enforces the absolute tolerance on the residual norm, so that the convergence condition
    //is applied in the same manner as the other KEMField linear algebra solvers
    PetscReal relative_tol = 0.0;
    PetscReal absolute_tol = this->fTolerance;
    ierr = KSPSetTolerances(ksp, relative_tol, absolute_tol, PETSC_DEFAULT, PETSC_DEFAULT);

    /*
      Set runtime options, e.g.,
      -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
      These options will override those specified above as long as
      KSPSetFromOptions() is called _after_ any other customization
      routines.
    */
    ierr = KSPSetFromOptions(ksp);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Solve the linear system
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
      Solve linear system
    */
    ierr = KSPSolve(ksp, b_, x_);

    // /*
    //   View solver info; we could instead use the option -ksp_view to
    //   print this info to the screen at the conclusion of KSPSolve().
    // */
    ierr = KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Apply solution and clean up
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    VecScatter ctx;
    Vec x_gathered;
    VecScatterCreateToZero(x_, &ctx, &x_gathered);
    VecScatterBegin(ctx, x_, x_gathered, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(ctx, x_, x_gathered, INSERT_VALUES, SCATTER_FORWARD);

    if (KMPIInterface::GetInstance()->GetProcess() == 0) {
        const PetscScalar* x_solved;
        VecGetArrayRead(x_gathered, &x_solved);

        for (int j = 0; j < n; j++)
            x[j] = x_solved[j];
    }

    ierr = VecScatterDestroy(&ctx);
    ierr = VecDestroy(&x_gathered);

    /*
      Free work space.  All PETSc objects should be destroyed when they
      are no longer needed.
    */
    ierr = VecDestroy(&x_);
    ierr = VecDestroy(&b_);
    ierr = MatDestroy(&A_);
    ierr = KSPDestroy(&ksp);

    //shut up the compiler
    (void) ierr;
}
}  // namespace KEMField

#endif /* KPETSCSOLVER_DEF */
