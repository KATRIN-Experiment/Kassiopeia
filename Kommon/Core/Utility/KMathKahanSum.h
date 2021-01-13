/**
 * @file KMathKahanSum.h
 *
 * @date 03.09.2013
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */

#ifndef KMATHKAHANSUM_H_
#define KMATHKAHANSUM_H_

namespace katrin
{

/**
    @brief Kahan summation algorithm
    The Kahan summation algorithm reduces the numerical error obtained with standard
    sequential sum.
*/
template<class FloatT = double> class KMathKahanSum
{
  public:
    KMathKahanSum(FloatT initialValue = 0.0) : fSum(0.0), fCompensation(0.0)
    {
        Add(initialValue);
    }
    virtual ~KMathKahanSum() = default;

    KMathKahanSum& Add(FloatT summand);
    KMathKahanSum& Subtract(FloatT summand)
    {
        return Add(-summand);
    }
    FloatT Result() const
    {
        return fSum;
    }

    KMathKahanSum& operator+=(FloatT summand)
    {
        return Add(summand);
    }
    KMathKahanSum& operator-=(FloatT summand)
    {
        return Subtract(summand);
    }
    //    KMathKahanSum& operator() (FloatT summand) { return Add(summand); }
    operator FloatT() const
    {
        return Result();
    }

    KMathKahanSum& Reset(FloatT initialValue = 0.0)
    {
        fSum = fCompensation = 0.0;
        return Add(initialValue);
    }

  protected:
    FloatT fSum;
    FloatT fCompensation;
};

#if _MSC_VER > 1400
#pragma float_control(push)
#pragma float_control(precise, on)
#endif


#define KAHANSUM_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

template<class FloatT>
inline KMathKahanSum<FloatT>&
#if KAHANSUM_GCC_VERSION > 40305
    __attribute__((__optimize__("no-associative-math")))
#endif
    KMathKahanSum<FloatT>::Add(FloatT summand)
{
    const FloatT myTmp1 = summand - fCompensation;
    const FloatT myTmp2 = fSum + myTmp1;
    fCompensation = (myTmp2 - fSum) - myTmp1;
    fSum = myTmp2;
    return *this;
}

#if _MSC_VER > 1400
#pragma float_control(pop)
#endif

template<class FloatT>
inline KMathKahanSum<FloatT> operator+(const KMathKahanSum<FloatT>& s1, const KMathKahanSum<FloatT>& s2)
{
    return KMathKahanSum<FloatT>(s1).Add(s2.Result());
}

template<class FloatT> inline KMathKahanSum<FloatT> operator+(const KMathKahanSum<FloatT>& s1, FloatT s2)
{
    return KMathKahanSum<FloatT>(s1).Add(s2);
}

template<class FloatT>
inline KMathKahanSum<FloatT> operator-(const KMathKahanSum<FloatT>& s1, const KMathKahanSum<FloatT>& s2)
{
    return KMathKahanSum<FloatT>(s1).Subtract(s2.Result());
}

template<class FloatT> inline KMathKahanSum<FloatT> operator-(const KMathKahanSum<FloatT>& s1, FloatT s2)
{
    return KMathKahanSum<FloatT>(s1).Subtract(s2);
}

} /* namespace katrin */
#endif /* KMATHKAHANSUM_H_ */
