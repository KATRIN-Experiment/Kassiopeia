#ifndef __KBlocKBlockJacobiPreconditioner_H__
#define __KBlocKBlockJacobiPreconditioner_H__

#include "KFMLinearAlgebraDefinitions.hh"
#include "KFMMatrixOperations.hh"
#include "KFMMatrixVectorOperations.hh"
#include "KFMVectorOperations.hh"
#include "KPreconditioner.hh"
#include "KSimpleVector.hh"

#include <vector>

namespace KEMField
{

/**
*
*@file KBlocKBlockJacobiPreconditioner.hh
*@class KBlocKBlockJacobiPreconditioner
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Aug 22 13:00:57 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename ValueType> class KBlockJacobiPreconditioner : public KPreconditioner<ValueType>
{
  public:
    KBlockJacobiPreconditioner(const KSquareMatrix<ValueType>& A,
                               const std::vector<const std::vector<unsigned int>*>* block_index_lists) :
        fDimension(A.Dimension()),
        fZero(0)
    {
        fBlockIndexLists.clear();
        for (unsigned int i = 0; i < block_index_lists->size(); i++) {
            std::vector<unsigned int> block_list;
            block_list.clear();
            for (unsigned int j = 0; j < block_index_lists->at(i)->size(); j++) {
                block_list.push_back(block_index_lists->at(i)->at(j));
            }

            if (block_list.size() != 0) {
                fBlockIndexLists.push_back(block_list);
            }
        }

        fNBlocks = fBlockIndexLists.size();

        //now we allocate and fill the matrices for each block
        for (unsigned int i = 0; i < fNBlocks; i++) {
            unsigned int block_size = fBlockIndexLists[i].size();
            kfm_matrix* block = kfm_matrix_alloc(block_size, block_size);

            for (unsigned int row = 0; row < block_size; row++) {
                for (unsigned int col = 0; col < block_size; col++) {
                    kfm_matrix_set(block, row, col, A(fBlockIndexLists[i][row], fBlockIndexLists[i][col]));
                }
            }
            fBlocks.push_back(block);
        }

        //now we invert each block using SVD
        for (unsigned int i = 0; i < fNBlocks; i++) {
            kfm_matrix* block = fBlocks[i];
            unsigned int size = block->size1;

            kfm_matrix* U = kfm_matrix_calloc(size, size);
            kfm_vector* S = kfm_vector_calloc(size);
            kfm_matrix* V = kfm_matrix_calloc(size, size);

            kfm_matrix_svd(block, U, S, V);

            //now we compute the inverse of the block from the SVD by
            //block^-1 = [V*diag(S)^{-1}*U^{T}]

            //now we apply the inverse of diag(S) to the columns of U
            //with the exception that if a singular value is zero then we apply zero
            //we assume anything less than KFM_EPSILON*norm(S) to be essentially zero (singular values should all be positive)
            double s, inv_s;
            double norm_s = kfm_vector_norm(S);
            for (unsigned int row = 0; row < size; row++) {
                s = kfm_vector_get(S, row);

                if (s > KFM_EPSILON * norm_s) {
                    inv_s = (1.0 / s);
                    //now scale the column of U by inv_s (this is the row of U^T)
                    for (unsigned int col = 0; col < size; col++) {
                        kfm_matrix_set(block, row, col, kfm_matrix_get(block, row, col) * inv_s);
                    }
                }
                else {
                    //zero out this column
                    for (unsigned int col = 0; col < size; col++) {
                        kfm_matrix_set(block, row, col, 0.0);
                    }
                }
            }

            //now multiply V agains U^T and place in the current block
            kfm_matrix_multiply_with_transpose(false, true, V, U, block);

            //free scratch space
            kfm_matrix_free(U);
            kfm_matrix_free(V);
            kfm_vector_free(S);
        }
    };

    virtual ~KBlockJacobiPreconditioner()
    {
        //free the blocks
        for (unsigned int i = 0; i < fNBlocks; i++) {
            kfm_matrix_free(fBlocks[i]);
        }
    };

    virtual std::string Name()
    {
        return std::string("bjacobi");
    };

    virtual void Multiply(const KVector<ValueType>& x, KVector<ValueType>& y) const
    {
        for (unsigned int n = 0; n < fNBlocks; n++) {
            kfm_matrix* block = fBlocks[n];
            unsigned int size = block->size1;

            //allocate temp vectors...TODO avoid this allocation/deallocation
            kfm_vector* in = kfm_vector_alloc(size);
            kfm_vector* out = kfm_vector_alloc(size);

            //fill the input vector
            for (unsigned int i = 0; i < size; i++) {
                kfm_vector_set(in, i, x(fBlockIndexLists[n][i]));
            }

            //apply the block
            kfm_matrix_vector_product(fBlocks[n], in, out);

            //copy back the output
            for (unsigned int i = 0; i < size; i++) {
                y[fBlockIndexLists[n][i]] = kfm_vector_get(out, i);
            }

            //deallocate temp vectors
            kfm_vector_free(in);
            kfm_vector_free(out);
        }
    }

    virtual void MultiplyTranspose(const KVector<ValueType>& x, KVector<ValueType>& y) const
    {
        for (unsigned int n = 0; n < fNBlocks; n++) {
            kfm_matrix* block = fBlocks[n];
            unsigned int size = block->size1;

            //allocate temp vectors...TODO avoid this allocation/deallocation
            kfm_vector* in = kfm_vector_alloc(size);
            kfm_vector* out = kfm_vector_alloc(size);

            //fill the input vector
            for (unsigned int i = 0; i < size; i++) {
                kfm_vector_set(in, i, x(fBlockIndexLists[n][i]));
            }

            //apply the block
            kfm_matrix_transpose_vector_product(fBlocks[n], in, out);

            //copy back the output
            for (unsigned int i = 0; i < size; i++) {
                y[fBlockIndexLists[n][i]] = kfm_vector_get(out, i);
            }

            //deallocate temp vectors
            kfm_vector_free(in);
            kfm_vector_free(out);
        }
    }

    virtual bool IsStationary()
    {
        return true;
    };

    virtual unsigned int Dimension() const
    {
        return fDimension;
    };

    virtual const ValueType& operator()(unsigned int /*i*/, unsigned int /*j*/) const
    {
        //TODO implement this
        return fZero;
    }

  protected:
    unsigned int fDimension;
    unsigned int fNBlocks;
    std::vector<std::vector<unsigned int>> fBlockIndexLists;
    std::vector<kfm_matrix*> fBlocks;

    ValueType fZero;
};


}  // namespace KEMField

#endif /* __KBlocKBlockJacobiPreconditioner_H__ */
