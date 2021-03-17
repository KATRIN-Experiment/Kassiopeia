#ifndef KSQUAREMATRIX_DEF
#define KSQUAREMATRIX_DEF

#include "KMatrix.hh"

namespace KEMField
{
template<typename ValueType> class KSquareMatrix : public KMatrix<ValueType>
{
  public:
    KSquareMatrix() : KMatrix<ValueType>() {}
    ~KSquareMatrix() override = default;

    virtual unsigned int Dimension() const = 0;
    const ValueType& operator()(unsigned int, unsigned int) const override = 0;

    unsigned int Dimension(unsigned int) const override
    {
        return Dimension();
    }
};
}  // namespace KEMField

#endif /* KSQUAREMATRIX_DEF */
