/*
 * KMatrixPreconditioner.hh
 *
 *  Created on: 13 Aug 2015
 *      Author: wolfgang
 */

#ifndef KMATRIXPRECONDITIONER_HH_
#define KMATRIXPRECONDITIONER_HH_

#include "KPreconditioner.hh"

#include <memory>

namespace KEMField
{

/**
 *  Matrix Preconditioner allows any KSquareMatrix to be used as KPreconditioner.
 *  The KPreconditioner specific functions IsStationary and Name are implemented
 *  with default values.
 */

template<typename ValueType> class KMatrixPreconditioner : public KPreconditioner<ValueType>
{
  public:
    explicit KMatrixPreconditioner(std::shared_ptr<const KSquareMatrix<ValueType>> matrix);
    ~KMatrixPreconditioner() override = default;

    std::string Name() override
    {
        return "Created from unnamed matrix";
    }

    bool IsStationary() override
    {
        return false;
    }

    //from KSquareMatrix
    unsigned int Dimension() const override
    {
        return fMatrix->Dimension();
    }

    const ValueType& operator()(unsigned int i, unsigned int j) const override
    {
        return fMatrix->operator()(i, j);
    }

    //from KMatrix
    void Multiply(const KVector<ValueType>& x, KVector<ValueType>& y) const override
    {
        fMatrix->Multiply(x, y);
    }

    void MultiplyTranspose(const KVector<ValueType>& x, KVector<ValueType>& y) const override
    {
        fMatrix->MultiplyTranspose(x, y);
    }

  private:
    std::shared_ptr<const KSquareMatrix<ValueType>> fMatrix;
};

template<typename ValueType>
KMatrixPreconditioner<ValueType>::KMatrixPreconditioner(std::shared_ptr<const KSquareMatrix<ValueType>> matrix) :
    fMatrix(matrix)
{}

} /* namespace KEMField */

#endif /* KMATRIXPRECONDITIONER_HH_ */
