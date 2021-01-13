#ifndef KVECTOR_DEF
#define KVECTOR_DEF

#include <cmath>

namespace KEMField
{
template<typename ValueType> class KVector
{
  public:
    KVector() = default;
    virtual ~KVector() = default;

    virtual const ValueType& operator()(unsigned int) const = 0;
    virtual ValueType& operator[](unsigned int) = 0;

    void Multiply(double alpha, KVector<ValueType>& y) const;

    virtual void operator*=(const ValueType& alpha);
    virtual void operator+=(const KVector<ValueType>& aVector);
    virtual void operator-=(const KVector<ValueType>& aVector);

    virtual unsigned int Dimension() const = 0;

    virtual const ValueType& InfinityNorm() const
    {
        static ValueType infinityNorm;
        infinityNorm = 0.;
        for (unsigned int i = 0; i < Dimension(); i++)
            if (KVector::abs(this->operator()(i)) > infinityNorm)
                infinityNorm = KVector::abs(this->operator()(i));
        return infinityNorm;
    }

    // template specializations necessary to avoid compiler warnings for unsigned types
    static ValueType abs(ValueType argument)
    {
        return std::abs(argument);
    }

    virtual void Fill(ValueType x);
};

template<typename ValueType> void KVector<ValueType>::Multiply(double alpha, KVector<ValueType>& y) const
{
    // Computes vector y in the equation alpha*x = y
    for (unsigned int i = 0; i < Dimension(); i++)
        y[i] = alpha * this->operator()(i);
}

template<typename ValueType> void KVector<ValueType>::operator*=(const ValueType& alpha)
{
    for (unsigned int i = 0; i < Dimension(); i++)
        this->operator[](i) *= alpha;
}

template<typename ValueType> void KVector<ValueType>::operator+=(const KVector<ValueType>& aVector)
{
    for (unsigned int i = 0; i < Dimension(); i++)
        this->operator[](i) += aVector(i);
}

template<typename ValueType> void KVector<ValueType>::operator-=(const KVector<ValueType>& aVector)
{
    for (unsigned int i = 0; i < Dimension(); i++)
        this->operator[](i) -= aVector(i);
}

template<> inline unsigned int KVector<unsigned int>::abs(unsigned int argument)
{
    return argument;
}

template<typename ValueType> void KVector<ValueType>::Fill(ValueType x)
{
    for (unsigned int i = 0; i < Dimension(); i++)
        this->operator[](i) = x;
}

}  // namespace KEMField

#endif /* KVECTOR_DEF */
