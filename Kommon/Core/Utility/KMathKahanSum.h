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
class KMathKahanSum
{
public:
    KMathKahanSum(double initialValue = 0.0) : fSum(0.0), fCompensation(0.0) { Add(initialValue); }
    virtual ~KMathKahanSum() { }

    KMathKahanSum& Add(double summand);
    KMathKahanSum& Subtract(double summand) { return Add(-summand); }
    double Result() const { return fSum; }

    KMathKahanSum& operator+= (double summand) { return Add(summand); }
    KMathKahanSum& operator-= (double summand) { return Subtract(summand); }
    KMathKahanSum& operator() (double summand) { return Add(summand); }
    operator double() const { return Result(); }

    KMathKahanSum& Reset(double initialValue = 0.0) { fSum = fCompensation = 0.0; return Add(initialValue); }

protected:
    double fSum;
    double fCompensation;
};

#if _MSC_VER > 1400
# pragma float_control(push)
# pragma float_control(precise, on)
#endif


#define KAHANSUM_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

inline KMathKahanSum&
#if KAHANSUM_GCC_VERSION > 40305
__attribute__((__optimize__("no-associative-math")))
#endif
KMathKahanSum::Add(double summand)
{
    const double myTmp1 = summand - fCompensation;
    const double myTmp2 = fSum + myTmp1;
    fCompensation = (myTmp2 - fSum) - myTmp1;
    fSum = myTmp2;
    return *this;
}

#if _MSC_VER > 1400
# pragma float_control(pop)
#endif

inline KMathKahanSum operator+ (const KMathKahanSum& s1, const KMathKahanSum& s2) {
    return KMathKahanSum( s1 ).Add( s2.Result() );
}

inline KMathKahanSum operator+ (const KMathKahanSum& s1, double s2) {
    return KMathKahanSum( s1 ).Add( s2 );
}

inline KMathKahanSum operator- (const KMathKahanSum& s1, const KMathKahanSum& s2) {
    return KMathKahanSum( s1 ).Subtract( s2.Result() );
}

inline KMathKahanSum operator- (const KMathKahanSum& s1, double s2) {
    return KMathKahanSum( s1 ).Subtract( s2 );
}

} /* namespace katrin */
#endif /* KMATHKAHANSUM_H_ */
