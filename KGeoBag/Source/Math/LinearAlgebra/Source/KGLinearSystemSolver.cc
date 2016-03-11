#include "KGLinearSystemSolver.hh"

namespace KGeoBag
{


KGLinearSystemSolver::KGLinearSystemSolver(unsigned int dim):fDim(dim)
{
    fA = kg_matrix_alloc(fDim, fDim);
    fU = kg_matrix_alloc(fDim, fDim);
    fX = kg_vector_alloc(fDim);
    fB = kg_vector_alloc(fDim);

    fV = kg_matrix_alloc(fDim,fDim);
    fS = kg_vector_alloc(fDim);
    fWork = kg_vector_alloc(fDim);


    fDimSize[0] = fDim;
    fDimSize[1] = fDim;
}

KGLinearSystemSolver::~KGLinearSystemSolver()
{
    kg_matrix_free(fA);
    kg_matrix_free(fU);
    kg_vector_free(fX);
    kg_vector_free(fB);

    kg_matrix_free(fV);
    kg_vector_free(fS);
    kg_vector_free(fWork);
}


void
KGLinearSystemSolver::SetMatrix(const double* mx) //expects row major ordering, and an array of size fDim*fDim
{
    unsigned int index[2];
    unsigned int offset;
    for(unsigned int row=0; row<fDim; row++)
    {
        index[0] = row;
        for(unsigned int col=0; col<fDim; col++)
        {
            index[1] = col;
            offset = KGArrayMath::OffsetFromRowMajorIndex<2>(fDimSize, index);
            kg_matrix_set(fA, row, col, mx[offset]);
        }
    }
}

void
KGLinearSystemSolver::SetMatrixElement(unsigned int row, unsigned int col, const double& val)
{
    kg_matrix_set(fA, row, col, val);
}

void
KGLinearSystemSolver::SetBVector(const double* vec)
{
    for(unsigned int i=0; i<fDim; i++)
    {
        kg_vector_set(fB, i, vec[i]);
    }
}

void
KGLinearSystemSolver::SetBVectorElement(unsigned int index, const double& val)
{
    kg_vector_set(fB, index, val);
}

void
KGLinearSystemSolver::Reset()
{
    for(unsigned int row=0; row<fDim; row++)
    {
        kg_vector_set(fB, row, 0.);
        kg_vector_set(fX, row, 0.);
        for(unsigned int col=0; col<fDim; col++)
        {
            kg_matrix_set(fA, row, col, 0.);
        }
    }
}


void
KGLinearSystemSolver::Solve()
{
    //SVD decompose and solve...this is much more robust than LU decomp
    kg_matrix_svd(fA, fU, fS, fV);
    kg_matrix_svd_solve(fU, fS, fV, fB, fX);
}


void
KGLinearSystemSolver::GetXVector(double* vec) const
{
    for(unsigned int i=0; i<fDim; i++)
    {
       vec[i] = kg_vector_get(fX, i);
    }
}

double
KGLinearSystemSolver::GetXVectorElement(unsigned int i) const
{
    return kg_vector_get(fX, i);
}



}
