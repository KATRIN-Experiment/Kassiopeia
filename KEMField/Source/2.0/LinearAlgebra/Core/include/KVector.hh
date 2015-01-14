#ifndef KVECTOR_DEF
#define KVECTOR_DEF

#include <cmath>

namespace KEMField
{
  template <typename ValueType>
  class KVector
  {
  public:
    KVector() {}
    virtual ~KVector() {}

    virtual const ValueType& operator()(unsigned int) const = 0;
    virtual ValueType& operator[](unsigned int) = 0;

  void Multiply(double alpha,KVector<ValueType>& y) const;

    virtual void operator*=(const ValueType& alpha);
    virtual void operator+=(const KVector<ValueType>& aVector);
    virtual void operator-=(const KVector<ValueType>& aVector);

    virtual unsigned int Dimension() const = 0;

    virtual const ValueType& InfinityNorm() const
    {
      static ValueType infinityNorm;
      infinityNorm = 0.;
      for (unsigned int i=0;i<Dimension();i++)
	if (fabs(this->operator()(i))>infinityNorm)
	  infinityNorm = fabs(this->operator()(i));
      return infinityNorm;
    }
  };

  template <typename ValueType>
  void KVector<ValueType>::Multiply(double alpha,
				    KVector<ValueType>& y) const
  {
    // Computes vector y in the equation alpha*x = y
    for (unsigned int i=0;i<Dimension();i++)
      y[i] = alpha*this->operator()(i);
  }

  template <typename ValueType>
  void KVector<ValueType>::operator*=(const ValueType& alpha)
  {
    for (unsigned int i=0;i<Dimension();i++)
      this->operator[](i)*=alpha;
  }

  template <typename ValueType>
  void KVector<ValueType>::operator+=(const KVector<ValueType>& aVector)
  {
    for (unsigned int i=0;i<Dimension();i++)
      this->operator[](i)+=aVector(i);
  }

  template <typename ValueType>
  void KVector<ValueType>::operator-=(const KVector<ValueType>& aVector)
  {
    for (unsigned int i=0;i<Dimension();i++)
      this->operator[](i)-=aVector(i);
  }
}

#endif /* KVECTOR_DEF */
