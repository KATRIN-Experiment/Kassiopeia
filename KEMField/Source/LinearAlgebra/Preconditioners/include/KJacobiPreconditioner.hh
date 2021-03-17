#ifndef KJacobiPreconditioner_HH__
#define KJacobiPreconditioner_HH__

#include "KPreconditioner.hh"
#include "KSimpleVector.hh"

namespace KEMField
{

/*
*
*@file KJacobiPreconditioner.hh
*@class KJacobiPreconditioner
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Jun  3 10:14:40 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ValueType> class KJacobiPreconditioner : public KPreconditioner<ValueType>
{
  public:
    KJacobiPreconditioner(const KSquareMatrix<ValueType>& A) : fDimension(A.Dimension()), fZero(0)
    {
        fInverseDiagonal.resize(fDimension);

        //compute the inverse of the diagonal
        for (unsigned int i = 0; i < fDimension; i++) {
            fInverseDiagonal[i] = 1.0 / A(i, i);
        }
    };

    ~KJacobiPreconditioner() override = default;
    ;

    std::string Name() override
    {
        return std::string("jacobi");
    };

    void Multiply(const KVector<ValueType>& x, KVector<ValueType>& y) const override
    {
        for (unsigned int i = 0; i < fDimension; i++) {
            y[i] = fInverseDiagonal(i) * x(i);
        }
    }

    void MultiplyTranspose(const KVector<ValueType>& x, KVector<ValueType>& y) const override
    {
        //copy x into y
        for (unsigned int i = 0; i < fDimension; i++) {
            y[i] = fInverseDiagonal(i) * x(i);
        }
    }

    bool IsStationary() override
    {
        return true;
    };

    unsigned int Dimension() const override
    {
        return fDimension;
    };

    const ValueType& operator()(unsigned int i, unsigned int j) const override
    {
        if (i == j) {
            return fInverseDiagonal(i);
        };

        return fZero;
    }

  protected:
    unsigned int fDimension;
    KSimpleVector<ValueType> fInverseDiagonal;
    ValueType fZero;
};


}  // namespace KEMField


#endif /* KJacobiPreconditioner_H__ */
