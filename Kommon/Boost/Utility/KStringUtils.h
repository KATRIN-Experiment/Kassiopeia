/**
 * @file
 * Contains katrin::KStringUtils
 *
 * @date Created on: 09.02.2012
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */

#ifndef KSTRINGUTILS_H_
#define KSTRINGUTILS_H_

#include "KBaseStringUtils.h"
#include "KRandom.h"

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <type_traits>

// clang-format off
// C++ container types
#include <utility>
#include <array>
#include <vector>
#include <deque>
#include <forward_list>
#include <list>
#include <stack>
#include <queue>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
// clang-format on

namespace katrin
{

/**
 * Contains static functions for joining, exploding and serializing string sequences.
 */
class KStringUtils
{
  public:
    KStringUtils() = delete;

    static bool IsNumeric(const char& character);
    static bool IsNumeric(const std::string& string);

    //    static bool MatchWildcardString(std::string pattern, std::string input);

    template<typename Range1T, typename Range2T> static bool Contains(const Range1T&, const Range2T&);

    template<typename Range1T, typename Range2T>
    static bool ContainsOneOf(const Range1T&, std::initializer_list<Range2T>);

    template<typename Range1T, typename Range2T> static bool IContains(const Range1T&, const Range2T&);

    template<typename Range1T, typename Range2T>
    static bool IContainsOneOf(const Range1T&, std::initializer_list<Range2T>);

    template<typename Range1T, typename Range2T> static bool StartsWith(const Range1T&, const Range2T&);

    template<typename Range1T, typename Range2T> static bool IStartsWith(const Range1T&, const Range2T&);

    template<typename Range1T, typename Range2T> static size_t Distance(const Range1T&, const Range2T&);

    template<typename Range1T, typename Range2T> static size_t IDistance(const Range1T&, const Range2T&);

    template<typename Range1T, typename Range2T> static float Similarity(const Range1T&, const Range2T&);

    template<typename Range1T, typename Range2T> static float ISimilarity(const Range1T&, const Range2T&);

    template<class NumberT> static std::string GroupDigits(NumberT input, const std::string& sep = ",");
    static std::string GroupDigits(std::string input, const std::string& sep = ",");

    static std::string RandomAlphaNum(size_t length);

    static void HexDump(const void* data, size_t length, std::ostream& os = std::cout);
    template<class SequenceT>
    static void HexDump(const SequenceT& data, std::ostream& os = std::cout);

  private:
    template<typename Range1T, typename Range2T> static size_t LevenshteinDistance(const Range1T&, const Range2T&);
};

template<typename Range1T, typename Range2T>
size_t KStringUtils::LevenshteinDistance(const Range1T& r1, const Range2T& r2)
{
    /* Excerpt from https://en.wikipedia.org/wiki/Levenshtein_distance
 *
 *  function LevenshteinDistance(char s[0..m-1], char t[0..n-1]):
 *      // create two work vectors of integer distances
 *      declare int v0[n + 1]
 *      declare int v1[n + 1]
 *
 *      // initialize v0 (the previous row of distances)
 *      // this row is A[0][i]: edit distance for an empty s
 *      // the distance is just the number of characters to delete from t
 *      for i from 0 to n:
 *          v0[i] = i
 *
 *      for i from 0 to m-1:
 *          // calculate v1 (current row distances) from the previous row v0
 *
 *          // first element of v1 is A[i+1][0]
 *          //   edit distance is delete (i+1) chars from s to match empty t
 *          v1[0] = i + 1
 *
 *          // use formula to fill in the rest of the row
 *          for j from 0 to n-1:
 *              // calculating costs for A[i+1][j+1]
 *              deletionCost := v0[j + 1] + 1
 *              insertionCost := v1[j] + 1
 *              if s[i] = t[j]:
 *                  substitutionCost := v0[j]
 *              else:
 *                  substitutionCost := v0[j] + 1;
 *
 *              v1[j + 1] := minimum(deletionCost, insertionCost, substitutionCost)
 *
 *          // copy v1 (current row) to v0 (previous row) for next iteration
 *          // since data in v1 is always invalidated, a swap without copy could be more efficient
 *          swap v0 with v1
 *      // after the last swap, the results of v1 are now in v0
 *      return v0[n]
 */

    const size_t m = std::end(r1) - std::begin(r1);  // instead of std::size(s1)
    const size_t n = std::end(r2) - std::begin(r2);  // instead of std::size(s2)
    std::vector<size_t> v0(n + 1);
    std::vector<size_t> v1(n + 1);

    for (size_t i = 0; i <= n; i++)
        v0[i] = i;

    for (size_t i = 0; i < m; i++) {
        v1[0] = i + 1;

        for (size_t j = 0; j < n; j++) {
            const size_t del = v0[j + 1] + 1;
            const size_t ins = v1[j] + 1;
            const size_t sub = (r1[i] == r2[j]) ? v0[j] : (v0[j] + 1);
            v1[j + 1] = std::min({del, ins, sub});
        }
        v0.swap(v1);
    }

    return v0[n];
}

inline bool KStringUtils::IsNumeric(const char& character)
{
    try {
        boost::lexical_cast<int>(character);
        return true;
    }
    catch (boost::bad_lexical_cast&) {
        return false;
    }
}

inline bool KStringUtils::IsNumeric(const std::string& string)
{
    try {
        boost::lexical_cast<double>(string);
        return true;
    }
    catch (boost::bad_lexical_cast&) {
        return false;
    }
}

template<typename Range1T, typename Range2T> inline bool KStringUtils::Contains(const Range1T& r1, const Range2T& r2)
{
    return boost::contains(r1, r2);
}

template<typename Range1T, typename Range2T>
inline bool KStringUtils::ContainsOneOf(const Range1T& r1, std::initializer_list<Range2T> l)
{
    for (const auto& r2 : l)
        if (boost::contains(r1, r2))
            return true;
    return false;
}

template<typename Range1T, typename Range2T> inline bool KStringUtils::IContains(const Range1T& r1, const Range2T& r2)
{
    return boost::icontains(r1, r2);
}

template<typename Range1T, typename Range2T>
inline bool KStringUtils::IContainsOneOf(const Range1T& r1, std::initializer_list<Range2T> l)
{
    for (const auto& r2 : l)
        if (boost::icontains(r1, r2))
            return true;
    return false;
}

template<typename Range1T, typename Range2T> inline bool KStringUtils::StartsWith(const Range1T& r1, const Range2T& r2)
{
    return boost::starts_with(r1, r2);
}

template<typename Range1T, typename Range2T> inline bool KStringUtils::IStartsWith(const Range1T& r1, const Range2T& r2)
{
    return boost::istarts_with(r1, r2);
}

template<typename Range1T, typename Range2T> inline size_t KStringUtils::Distance(const Range1T& r1, const Range2T& r2)
{
    return LevenshteinDistance(r1, r2);
}

template<typename Range1T, typename Range2T> inline size_t KStringUtils::IDistance(const Range1T& r1, const Range2T& r2)
{
    return LevenshteinDistance(boost::to_upper_copy(r1), boost::to_upper_copy(r2));
}

template<typename Range1T, typename Range2T> inline float KStringUtils::Similarity(const Range1T& r1, const Range2T& r2)
{
    if (boost::equals(r1, r2))
        return 1.0;
    if (boost::empty(r1) || boost::empty(r2))
        return 0.0;
    return 1.0 / Distance(r1, r2);
}

template<typename Range1T, typename Range2T>
inline float KStringUtils::ISimilarity(const Range1T& r1, const Range2T& r2)
{
    if (boost::iequals(r1, r2))
        return 1.0;
    if (boost::empty(r1) || boost::empty(r2))
        return 0.0;
    return 1.0 / IDistance(r1, r2);
}

template<class NumberT> inline std::string KStringUtils::GroupDigits(NumberT input, const std::string& sep)
{
    return GroupDigits(boost::lexical_cast<std::string>(input), sep);
}

template<class SequenceT>
inline void KStringUtils::HexDump(const SequenceT& sequence, std::ostream& os)
{
    HexDump(sequence.data(), sequence.size() * sizeof(typename SequenceT::value_type), os);
}

}  // namespace katrin


namespace std
{

template<class T> inline std::ostream& operator<<(std::ostream& os, const boost::numeric::ublas::matrix<T>& matrix)
{
    using size_type = typename boost::numeric::ublas::matrix<T>::size_type;

    os << "[" << matrix.size1() << "," << matrix.size2() << "]";
    for (size_type r = 0; r < matrix.size1(); ++r) {
        os << std::endl << "(";
        for (size_type c = 0; c < matrix.size2(); ++c) {
            os.width(os.precision() + 7);
            os << matrix(r, c);
        }
        os << " )";
    }
    return os;
}

template<class T, class U>
inline std::ostream& operator<<(std::ostream& os, const boost::numeric::ublas::triangular_matrix<T, U>& matrix)
{
    using size_type = typename boost::numeric::ublas::matrix<T>::size_type;

    os << "[" << matrix.size1() << "," << matrix.size2() << "]";
    for (size_type r = 0; r < matrix.size1(); ++r) {
        os << std::endl << "(";
        for (size_type c = 0; c < matrix.size2(); ++c) {
            os.width(os.precision() + 7);
            os << matrix(r, c);
        }
        os << " )";
    }
    return os;
}

}  // namespace std

#endif /* KSTRINGUTILS_H_ */
