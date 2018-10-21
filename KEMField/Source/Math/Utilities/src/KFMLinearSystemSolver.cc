#include "KFMLinearSystemSolver.hh"

namespace KEMField
{


KFMLinearSystemSolver::KFMLinearSystemSolver(unsigned int dim):fDim(dim)
{
    fA = kfm_matrix_alloc(fDim, fDim);
    fU = kfm_matrix_alloc(fDim, fDim);
    fX = kfm_vector_alloc(fDim);
    fB = kfm_vector_alloc(fDim);

    fV = kfm_matrix_alloc(fDim,fDim);
    fS = kfm_vector_alloc(fDim);
    fWork = kfm_vector_alloc(fDim);


    fDimSize[0] = fDim;
    fDimSize[1] = fDim;
}

KFMLinearSystemSolver::~KFMLinearSystemSolver()
{
    kfm_matrix_free(fA);
    kfm_matrix_free(fU);
    kfm_vector_free(fX);
    kfm_vector_free(fB);

    kfm_matrix_free(fV);
    kfm_vector_free(fS);
    kfm_vector_free(fWork);
}


void
KFMLinearSystemSolver::SetMatrix(const double* mx) //expects row major ordering, and an array of size fDim*fDim
{
    unsigned int index[2];
    unsigned int offset;
    for(unsigned int row=0; row<fDim; row++)
    {
        index[0] = row;
        for(unsigned int col=0; col<fDim; col++)
        {
            index[1] = col;
            offset = KFMArrayMath::OffsetFromRowMajorIndex<2>(fDimSize, index);
            kfm_matrix_set(fA, row, col, mx[offset]);
        }
    }
}

void
KFMLinearSystemSolver::SetMatrixElement(unsigned int row, unsigned int col, const double& val)
{
    kfm_matrix_set(fA, row, col, val);
}

void
KFMLinearSystemSolver::SetBVector(const double* vec)
{
    for(unsigned int i=0; i<fDim; i++)
    {
        kfm_vector_set(fB, i, vec[i]);
    }
}

void
KFMLinearSystemSolver::SetBVectorElement(unsigned int index, const double& val)
{
    kfm_vector_set(fB, index, val);
}

void
KFMLinearSystemSolver::Reset()
{
    for(unsigned int row=0; row<fDim; row++)
    {
        kfm_vector_set(fB, row, 0.);
        kfm_vector_set(fX, row, 0.);
        for(unsigned int col=0; col<fDim; col++)
        {
            kfm_matrix_set(fA, row, col, 0.);
        }
    }
}


void
KFMLinearSystemSolver::Solve()
{
    //SVD decompose and solve...this is much more robust than LU decomp
    kfm_matrix_svd(fA, fU, fS, fV);
    kfm_matrix_svd_solve(fU, fS, fV, fB, fX);
}


void
KFMLinearSystemSolver::GetXVector(double* vec) const
{
    for(unsigned int i=0; i<fDim; i++)
    {
       vec[i] = kfm_vector_get(fX, i);
    }
}

double
KFMLinearSystemSolver::GetXVectorElement(unsigned int i) const
{
    return kfm_vector_get(fX, i);
}



}
