/**
 * @file
 * Numeric utility functions, primarily for real numbers.
 * katrin::KNumeric
 *
 * @date   Created on: 09.02.2012
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */

#ifndef KNUMERIC_H_
#define KNUMERIC_H_

#include <limits>
#include <cmath>
#include <type_traits>

namespace katrin {

/**
 * Class containing static numerical utility functions.
 */
class KNumeric {
public:
    KNumeric() = delete;

    template <class T>
    static T Sign(const T& value) { return std::copysign(T(1.0), value); }

    template <class T>
    static int IntSign(const T& value) { return ( value == T(0) ) ? 0 : ( (value < T(0) ) ? -1 : 1 ); }

    template<class IntegerT>
    static bool IsOdd(const IntegerT& v) { return (v % 2 == 1); }
    template<class IntegerT>
    static bool IsEven(const IntegerT& v) { return (v % 2 == 0); }

    template<class ReturnT, class InputIntegerT>
    static ReturnT IfOddThenElse(const InputIntegerT& input, const ReturnT& thenValue, const ReturnT& elseValue)
    { return (input % 2 == 1) ? thenValue : elseValue; }
    template<class ReturnT, class InputIntegerT>
    static ReturnT IfEvenThenElse(const InputIntegerT& input, const ReturnT& thenValue, const ReturnT& elseValue)
    { return (input % 2 == 0) ? thenValue : elseValue; }

    template<class T>
    static void Limit(T& input, const T& min, const T& max);

    /**
     * Check whether two float values differ by a given epsilon.
     * @param a
     * @param b
     * @param epsilon
     * @return <code>return fabs(a - b) <= ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);</code>
     */
    template<class T>
    static bool ApproximatelyEqual(T a, T b, T epsilon);
    template<class T>
    static bool ApproximatelyLessOrEqual(T a, T b, T epsilon) { return a < b || ApproximatelyEqual(a, b, epsilon); }
    template<class T>
    static bool ApproximatelyGreaterOrEqual(T a, T b, T epsilon) { return a > b || ApproximatelyEqual(a, b, epsilon); }

    /**
     * Check whether two float values differ by a given epsilon.
     * @param a
     * @param b
     * @param epsilon
     * @return <code>return fabs(a - b) <= ( (fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * epsilon);</code>
     */
    template<class T>
    static bool EssentiallyEqual(T a, T b, T epsilon);
    template<class T>
    static bool EssentiallyLessOrEqual(T a, T b, T epsilon) { return a < b || EssentiallyEqual(a, b, epsilon); }
    template<class T>
    static bool EssentiallyGreaterOrEqual(T a, T b, T epsilon) { return a > b || EssentiallyEqual(a, b, epsilon); }

    template<class T>
    static bool DefinitelyGreaterThan(T a, T b, T epsilon);

    template<class T>
    static bool DefinitelyLessThan(T a, T b, T epsilon);

    template<typename T>
    static typename std::make_signed<T>::type Cast2Signed(const T& value) { return value; }
};

template<class T>
inline void KNumeric::Limit(T& input, const T& min, const T& max)
{
    if (min > max)
        return;
    else if (input < min)
        input = min;
    else if (input > max)
        input = max;
}

template<class T>
inline bool KNumeric::ApproximatelyEqual(T a, T b, T epsilon)
{
    return fabs(a - b) <= ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

template<class T>
inline bool KNumeric::EssentiallyEqual(T a, T b, T epsilon)
{
    return fabs(a - b) <= ( (fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

template<class T>
inline bool KNumeric::DefinitelyGreaterThan(T a, T b, T epsilon)
{
    return (a - b) > ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

template<class T>
inline bool KNumeric::DefinitelyLessThan(T a, T b, T epsilon)
{
    return (b - a) > ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

}

#endif /* KNUMERIC_H_ */
