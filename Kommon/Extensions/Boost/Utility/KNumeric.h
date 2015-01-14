/**
 * @file
 * Numeric utility functions for float values.
 * katrin::KNumeric
 *
 * @date   Created on: 09.02.2012
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */

#ifndef KNUMERIC_H_
#define KNUMERIC_H_

#include <limits>
#include <vector>

#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/optional.hpp>
#include <boost/type_traits/make_signed.hpp>
#include <boost/lexical_cast.hpp>

namespace katrin {

/**
 * Class containing static numerical utility functions.
 */
class KNumeric {
public:
    template<class T>
    static T NaN()             { return std::numeric_limits<T>::quiet_NaN(); }
    static double NaN()      { return std::numeric_limits<double>::quiet_NaN(); }

    template<class T>
    static T Infinity()        { return std::numeric_limits<T>::infinity(); }
    static double Infinity() { return std::numeric_limits<double>::infinity(); }

    template<class T>
    static T Epsilon()        { return std::numeric_limits<T>::epsilon(); }
    static double Epsilon() { return std::numeric_limits<double>::epsilon(); }

    template<class T>
    static T Min()         { return std::numeric_limits<T>::min(); }
    static double Min()  { return std::numeric_limits<double>::min(); }

    template<class T>
    static T Max()         { return std::numeric_limits<T>::max(); }
    static double Max()  { return std::numeric_limits<double>::max(); }

    template <class T>
    static T Sign(const T& value) { return ( value == T(0) ) ? T(0) : ( (value < T(0) ) ? T(-1) : T(1) ); }

    template <class T>
    static int IntSign(const T& value) { return ( value == T(0) ) ? 0 : ( (value < T(0) ) ? -1 : 1 ); }

    template<class T>
    static bool IsNaN(const T& v) { return boost::math::isnan(v); }

    template<class T>
    static bool IsInf(const T& v) { return boost::math::isinf(v); }

    /**
     * Check if a number is finite.
     * @param v
     * @return True if the value is neither NaN, nor infinite.
     */
    template<class T>
    static bool IsFinite(const T& v) { return boost::math::isfinite(v); }

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

    /**
     * Return the input value constrained to a certain range.
     * @param input
     * @param lowerBound
     * @param upperBound
     * @return
     */
    template<class T>
    static T Limit(const T& input, const boost::optional<T>& lowerBound = boost::none,
        const boost::optional<T>& upperBound = boost::none);

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

    /**
     * Check whether two floating-point numbers differ by a given <i>unit of least precision</i> (ULP).
     * @param A
     * @param B
     * @param maxUlps
     * @return
     */
    template<class T>
    static bool ApproximatelyEqualUlp(T A, T B, unsigned int maxUlp);

    template<typename T>
    static typename boost::make_signed<T>::type Cast2Signed(const T& value) { return value; }

    template<typename Output, typename Input>
    static Output LexicalCast(const Input& input);

    template<typename Output, typename Input>
    static Output LexicalCast(const Input& input, const Output& defaultValue);
};

template<class T>
inline T KNumeric::Limit(const T& input, const boost::optional<T>& lowerBound, const boost::optional<T>& upperBound)
{
    if (lowerBound && input < lowerBound.get())
        return lowerBound.get();
    else if (upperBound && input > upperBound.get())
        return upperBound.get();
    else
        return input;
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

template<class T>
inline bool KNumeric::ApproximatelyEqualUlp(T A, T B, unsigned int maxUlp)
{
    return (fabs(boost::math::float_distance(A, B)) <= (double) maxUlp);
}

template<typename Output, typename Input>
inline Output KNumeric::LexicalCast(const Input& input)
{
    return boost::lexical_cast<Output>(input);
}

template<typename Output, typename Input>
inline Output KNumeric::LexicalCast(const Input& input, const Output& defaultValue)
{
    try {
        return boost::lexical_cast<Output>(input);
    }
    catch (boost::bad_lexical_cast&) {
        return defaultValue;
    }
}

}


template<class T>
inline std::vector<T>& operator*= (std::vector<T>& v, T f)
{
    for (std::size_t i=0; i<v.size(); ++i)
        v[i] *= f;
    return v;
}

template<class T>
inline std::vector<T> operator* (const std::vector<T>& v, T f)
{
    std::vector<T> result(v);
    result *= f;
    return result;
}

template<class T>
inline std::vector<T> operator* (T f, const std::vector<T>& v)
{
    return v*f;
}


#endif /* KNUMERIC_H_ */
