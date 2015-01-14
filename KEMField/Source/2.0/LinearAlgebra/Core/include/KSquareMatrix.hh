#ifndef KSQUAREMATRIX_DEF
#define KSQUAREMATRIX_DEF

#include "KMatrix.hh"

namespace KEMField
{
  template <typename ValueType>
  class KSquareMatrix : public KMatrix<ValueType>
  {
  public:
    KSquareMatrix() : KMatrix<ValueType>() {}
    virtual ~KSquareMatrix() {}

    virtual unsigned int Dimension() const = 0;
    virtual const ValueType& operator()(unsigned int,unsigned int) const = 0;

    unsigned int Dimension(unsigned int) const { return Dimension(); }
  };
}

#endif /* KSQUAREMATRIX_DEF */
