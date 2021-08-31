/**
 * @file
 * Contains katrin::KStringUtils
 *
 * @date Created on: 09.02.2012
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */

#ifndef KSTRINGUTILS_H_
#define KSTRINGUTILS_H_

#include "KRandom.h"

#include <array>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <deque>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

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

    template<typename TargetT> static bool Convert(const std::string& string, TargetT& result);

    template<typename Range1T, typename Range2T> static bool Equals(const Range1T&, const Range2T&);

    template<typename Range1T, typename Range2T> static bool IEquals(const Range1T&, const Range2T&);

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

    static std::string Trim(const std::string& input);
    static std::string TrimLeft(const std::string& input);
    static std::string TrimRight(const std::string& input);

    template<class SequenceT, class SeparatorT>
    static std::ostream& Join(std::ostream& stream, const SequenceT& sequence, const SeparatorT& separator);

    template<class SequenceT, class SeparatorT>
    static std::string Join(const SequenceT& sequence, const SeparatorT& separator, int precision = -1);

    template<class KeyT, class ValueT, class SeparatorT>
    static std::ostream& Join(std::ostream& stream, const std::map<KeyT, ValueT>& map, const SeparatorT& separator);

    template<class KeyT, class ValueT, class SeparatorT>
    static std::string Join(const std::map<KeyT, ValueT>& map, const SeparatorT& separator, int precision = -1);

    template<class OutputT = std::string>
    static int Split(const std::string& input, const std::string& delimiters, std::vector<OutputT>& output);

    template<class OutputT = std::string>
    static std::vector<OutputT> Split(const std::string& input, const std::string& delimiters);

    template<class OutputT = std::string>
    static int SplitBySingleDelim(const std::string& input, const std::string& delimiter, std::vector<OutputT>& output);

    template<class OutputT = std::string>
    static std::vector<OutputT> SplitBySingleDelim(const std::string& input, const std::string& delimiter);

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

template<typename TargetT> inline bool KStringUtils::Convert(const std::string& string, TargetT& result)
{
    try {
        result = boost::lexical_cast<TargetT>(string);
        return true;
    }
    catch (boost::bad_lexical_cast&) {
        return false;
    }
}

template<typename Range1T, typename Range2T> inline bool KStringUtils::Equals(const Range1T& r1, const Range2T& r2)
{
    return boost::equals(r1, r2);
}

template<typename Range1T, typename Range2T> inline bool KStringUtils::IEquals(const Range1T& r1, const Range2T& r2)
{
    return boost::iequals(r1, r2);
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

inline std::string KStringUtils::Trim(const std::string& input)
{
    return boost::algorithm::trim_copy(input);
}

inline std::string KStringUtils::TrimLeft(const std::string& input)
{
    return boost::algorithm::trim_left_copy(input);
}

inline std::string KStringUtils::TrimRight(const std::string& input)
{
    return boost::algorithm::trim_right_copy(input);
}

template<class SequenceT, class SeparatorT>
inline std::ostream& KStringUtils::Join(std::ostream& stream, const SequenceT& sequence, const SeparatorT& separator)
{
    // Parse input
    auto itBegin = std::begin(sequence);
    auto itEnd = std::end(sequence);

    // Append first element
    if (itBegin != itEnd) {
        stream << *itBegin;
        ++itBegin;
    }
    for (; itBegin != itEnd; ++itBegin) {
        stream << boost::as_literal(separator);
        stream << *itBegin;
    }
    return stream;
}

template<class SequenceT, class SeparatorT>
inline std::string KStringUtils::Join(const SequenceT& sequence, const SeparatorT& separator, int precision)
{
    std::ostringstream result;
    if (precision >= 0)
        result.precision(precision);
    Join(result, sequence, separator);
    return result.str();
}

template<class KeyT, class ValueT, class SeparatorT>
inline std::ostream& KStringUtils::Join(std::ostream& stream, const std::map<KeyT, ValueT>& map,
                                        const SeparatorT& separator)
{
    // Parse input
    auto itBegin = std::begin(map);
    auto itEnd = std::end(map);

    // Append first element
    if (itBegin != itEnd) {
        stream << "[" << itBegin->first << "] " << itBegin->second;
        itBegin++;
    }
    for (; itBegin != itEnd; ++itBegin) {
        stream << boost::as_literal(separator);
        stream << "[" << itBegin->first << "] " << itBegin->second;
    }
    return stream;
}

template<class KeyT, class ValueT, class SeparatorT>
inline std::string KStringUtils::Join(const std::map<KeyT, ValueT>& map, const SeparatorT& separator, int precision)
{
    std::ostringstream result;
    if (precision >= 0)
        result.precision(precision);
    Join(result, map, separator);
    return result.str();
}

template<class OutputT>
inline int KStringUtils::Split(const std::string& input, const std::string& delimiters, std::vector<OutputT>& output)
{
    std::vector<std::string> tokens;
    boost::split(tokens, input, boost::is_any_of(delimiters), boost::token_compress_on);
    output.clear();
    int returnValue = 0;

    for (std::string& token : tokens) {
        boost::trim(token);
        if (boost::is_floating_point<OutputT>::value)
            boost::replace_all(token, ",", ".");
        if (token.empty())
            continue;
        try {
            output.push_back(boost::lexical_cast<OutputT>(token));
        }
        catch (boost::bad_lexical_cast& e) {
            returnValue = -1;
        }
    }

    if (output.empty() && !input.empty())
        returnValue = -1;
    else if (returnValue >= 0)
        returnValue = (int) output.size();

    return returnValue;
}

template<class OutputT>
inline std::vector<OutputT> KStringUtils::Split(const std::string& input, const std::string& delimiters)
{
    std::vector<OutputT> result;
    KStringUtils::Split(input, delimiters, result);
    return result;
}

template<class OutputT>
inline int KStringUtils::SplitBySingleDelim(const std::string& input, const std::string& delimiter,
                                            std::vector<OutputT>& output)
{
    std::vector<std::string> tokens;
    std::back_insert_iterator<std::vector<std::string>> tokenIt = std::back_inserter(tokens);

    using namespace boost::algorithm;
    typedef split_iterator<std::string::const_iterator> It;

    for (It iter = make_split_iterator(input, first_finder(delimiter, is_equal())); iter != It(); ++iter) {
        *(tokenIt++) = boost::copy_range<std::string>(*iter);
    }

    output.clear();
    int returnValue = 0;

    for (std::string& token : tokens) {
        boost::trim(token);
        if (boost::is_floating_point<OutputT>::value)
            boost::replace_all(token, ",", ".");
        if (token.empty())
            continue;
        try {
            output.push_back(boost::lexical_cast<OutputT>(token));
        }
        catch (boost::bad_lexical_cast& e) {
            returnValue = -1;
        }
    }

    if (output.empty() && !input.empty())
        returnValue = -1;
    else if (returnValue >= 0)
        returnValue = (int) output.size();

    return returnValue;
}

template<class OutputT>
inline std::vector<OutputT> KStringUtils::SplitBySingleDelim(const std::string& input, const std::string& delimiter)
{
    std::vector<OutputT> result;
    KStringUtils::SplitBySingleDelim(input, delimiter, result);
    return result;
}

template<class NumberT> inline std::string KStringUtils::GroupDigits(NumberT input, const std::string& sep)
{
    return GroupDigits(boost::lexical_cast<std::string>(input), sep);
}

inline std::string KStringUtils::GroupDigits(std::string str, const std::string& sep)
{
    boost::trim(str);
    size_t i = str.find_first_not_of("-+0123456789");

    if (i == std::string::npos)
        i = str.length();
    if (i < 4)
        return str;

    for (i -= 3; i > 0 && i < str.length(); i -= 3)
        str.insert(i, sep);

    return str;
}

inline std::string KStringUtils::RandomAlphaNum(size_t length)
{
    static const std::string alphanums = "0123456789"
                                         "abcdefghijklmnopqrstuvwxyz"
                                         "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    std::string result;
    result.reserve(length);
    while (length--)
        result += alphanums[KRandom::GetInstance().Uniform<size_t>(0, alphanums.length() - 1)];

    return result;
}

// adapted from
inline void KStringUtils::HexDump(const void *data, size_t length, std::ostream& os)
{
    size_t i;
    unsigned char linebuf[17];
    unsigned char *pc = (unsigned char*)data;

    // Process every byte in the data.
    for (i = 0; i < length; i++) {
        // Multiple of 16 means new line (with line offset).

        if ((i % 16) == 0) {
            // Just don't print ASCII for the zeroth line.
            if (i != 0)
                os << "  " << linebuf << "\n";

            // Output the offset.
            os << "  " << std::hex << std::setfill('0') << std::setw(4) << i;
        }

        // Now the hex code for the specific character.
        os << " " << std::hex << std::setfill('0') << std::setw(2) << pc[i];

        // And store a printable ASCII character for later.
        if ((pc[i] < 0x20) || (pc[i] > 0x7e)) {
            linebuf[i % 16] = '.';
        } else {
            linebuf[i % 16] = pc[i];
        }

        linebuf[(i % 16) + 1] = '\0';
    }

    // Pad out last line if not exactly 16 characters.
    while ((i % 16) != 0) {
        os << "   ";
        i++;
    }

    // And print the final ASCII bit.
    os << "  " << linebuf << "\n";
}

template<class SequenceT>
inline void KStringUtils::HexDump(const SequenceT& sequence, std::ostream& os)
{
    HexDump(sequence.data(), sequence.size() * sizeof(typename SequenceT::value_type), os);
}

}  // namespace katrin


namespace std
{

template<class ValueT> inline std::ostream& operator<<(std::ostream& os, const std::vector<ValueT>& v)
{
    os << "[" << v.size() << "]";
    if (!v.empty()) {
        os << "(";
        katrin::KStringUtils::Join(os, v, ", ");
        os << ")";
    }
    return os;
}

template<class ValueT> inline std::ostream& operator<<(std::ostream& os, const std::deque<ValueT>& v)
{
    os << "[" << v.size() << "]";
    if (!v.empty()) {
        os << "(";
        katrin::KStringUtils::Join(os, v, ", ");
        os << ")";
    }
    return os;
}

template<class ValueT> inline std::ostream& operator<<(std::ostream& os, const std::list<ValueT>& v)
{
    os << "[" << v.size() << "]";
    if (!v.empty()) {
        os << "(";
        katrin::KStringUtils::Join(os, v, ", ");
        os << ")";
    }
    return os;
}

template<class ValueT> inline std::ostream& operator<<(std::ostream& os, const std::set<ValueT>& v)
{
    os << "[" << v.size() << "]";
    if (!v.empty()) {
        os << "(";
        katrin::KStringUtils::Join(os, v, ", ");
        os << ")";
    }
    return os;
}

template<class KeyT, class ValueT> inline std::ostream& operator<<(std::ostream& os, const std::map<KeyT, ValueT>& v)
{
    os << "[" << v.size() << "]";
    if (!v.empty()) {
        os << "(";
        katrin::KStringUtils::Join(os, v, ", ");
        os << ")";
    }
    return os;
}

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

template<class InputT> inline std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<InputT>>& matrix)
{
    using size_type = typename std::vector<std::vector<InputT>>::size_type;

    const size_type nRows = matrix.size();
    const size_type nCols = (matrix.empty()) ? 0 : matrix.front().size();

    os << "[" << nRows << "," << nCols << "]";

    if (nCols * nRows == 0)
        return os;

    for (size_type r = 0; r < nRows; ++r) {
        os << std::endl << "(";
        for (size_type c = 0; c < matrix[r].size(); ++c) {
            os.width(os.precision() + 7);
            os << matrix[r][c];
        }
        os << " )";
    }
    return os;
}

template<class InputT, std::size_t S>
inline std::ostream& operator<<(std::ostream& os, const std::array<InputT, S>& arr)
{
    os << "[" << arr.size() << "]";
    if (!arr.empty()) {
        os << "(";
        katrin::KStringUtils::Join(os, arr, ", ");
        os << ")";
    }
    return os;
}

}  // namespace std

#endif /* KSTRINGUTILS_H_ */
