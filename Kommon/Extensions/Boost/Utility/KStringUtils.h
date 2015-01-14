/**
 * @file
 * Contains katrin::KStringUtils
 *
 * @date Created on: 09.02.2012
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */

#ifndef KSTRINGUTILS_H_
#define KSTRINGUTILS_H_

#include "KForeach.h"

#include <vector>
#include <list>
#include <set>
#include <string>
#include <iostream>
#include <iomanip>
#include <map>
#include <iterator>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/array.hpp>
#include <boost/type_traits.hpp>

namespace katrin {

/**
 * Contains static functions for joining, exploding and serializing string sequences.
 */
class KStringUtils {
public :

    static bool IsNumeric(const char& character);
    static bool IsNumeric(const std::string& string);

    template <class SequenceT, class SeparatorT>
    static std::ostream& Join(std::ostream& stream, const SequenceT& sequence, const SeparatorT& separator);

    template <class SequenceT, class SeparatorT>
    static std::string Join(const SequenceT& sequence, const SeparatorT& separator, int precision = -1);

    template <class KeyT, class ValueT>
    static std::ostream& Join(std::ostream& stream, const std::map<KeyT, ValueT>& map);

    template <class MapT>
    static std::string Join(const MapT& map);

    template <class OutputT>
    static int Split(const std::string& input, const std::string& delimiters, std::vector<OutputT>& output);

    template <class OutputT>
    static std::vector<OutputT> Split(const std::string& input, const std::string& delimiters);

    template <class OutputT>
    static int SplitBySingleDelim(const std::string& input, const std::string& delimiter, std::vector<OutputT>& output);

    template <class OutputT>
    static std::vector<OutputT> SplitBySingleDelim(const std::string& input, const std::string& delimiter);

    template <class NumberT>
    static std::string GroupDigits(NumberT input, const std::string& sep = ",");
    static std::string GroupDigits(std::string input, const std::string& sep = ",");
};

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

template <class SequenceT, class SeparatorT>
inline std::ostream& KStringUtils::Join(std::ostream& stream, const SequenceT& sequence, const SeparatorT& separator)
{
    typedef typename boost::range_const_iterator<SequenceT>::type InputIteratorT;

    // Parse input
    InputIteratorT itBegin = boost::begin(sequence);
    InputIteratorT itEnd = boost::end(sequence);

    // Append first element
    if(itBegin!=itEnd) {
        stream << *itBegin;
        ++itBegin;
    }

    for(;itBegin!=itEnd; ++itBegin) {
        stream << boost::as_literal(separator);
        stream << *itBegin;
    }

    return stream;
}

template <class SequenceT, class SeparatorT>
inline std::string KStringUtils::Join(const SequenceT& sequence, const SeparatorT& separator, int precision)
{
    std::ostringstream result;
    if (precision >= 0)
        result.precision(precision);
    Join(result, sequence, separator);
    return result.str();
}

template <class KeyT, class ValueT>
inline std::ostream& KStringUtils::Join(std::ostream& stream, const std::map<KeyT, ValueT>& map)
{
    typename std::map<KeyT, ValueT>::const_iterator it = map.begin();
    if (it != map.end()) {
        stream << "[" << it->first << "] " << it->second;
        it++;
    }
    for (; it != map.end(); ++it)
        stream << "\n[" << it->first << "] " << it->second;
    return stream;
}

template <class MapT>
inline std::string KStringUtils::Join(const MapT& map)
{
    std::ostringstream result;
    Join(result, map);
    return result.str();
}

template <class OutputT>
inline int KStringUtils::Split(const std::string& input, const std::string& delimiters, std::vector<OutputT>& output)
{
    std::vector<std::string> tokens;
    boost::split(tokens, input, boost::is_any_of(delimiters), boost::token_compress_on);
    output.clear();
    int returnValue = 0;

    foreach(std::string& token, tokens) {
        boost::trim(token);
        if (boost::is_floating_point<OutputT>::value)
            boost::replace_all(token, ",", ".");
        if (token.empty()) continue;
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

template <class OutputT>
inline std::vector<OutputT> KStringUtils::Split(const std::string& input, const std::string& delimiters)
{
    std::vector<OutputT> result;
    KStringUtils::Split(input, delimiters, result);
    return result;
}

template <class OutputT>
inline int KStringUtils::SplitBySingleDelim(const std::string& input, const std::string& delimiter, std::vector<OutputT>& output)
{
    std::vector<std::string> tokens;
    std::back_insert_iterator<std::vector<std::string> > tokenIt = std::back_inserter(tokens);

    using namespace boost::algorithm;
    typedef split_iterator<std::string::const_iterator> It;

    for (It iter = make_split_iterator(input, first_finder(delimiter, is_equal())); iter!=It(); ++iter) {
        *(tokenIt++) = boost::copy_range<std::string>(*iter);
    }

    output.clear();
    int returnValue = 0;

    foreach(std::string& token, tokens) {
        boost::trim(token);
        if (boost::is_floating_point<OutputT>::value)
            boost::replace_all(token, ",", ".");
        if (token.empty()) continue;
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

template <class OutputT>
inline std::vector<OutputT> KStringUtils::SplitBySingleDelim(const std::string& input, const std::string& delimiter)
{
    std::vector<OutputT> result;
    KStringUtils::SplitBySingleDelim(input, delimiter, result);
    return result;
}

template <class NumberT>
inline std::string KStringUtils::GroupDigits(NumberT input, const std::string& sep)
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

}

namespace std {

template <class ValueT>
inline std::ostream& operator<< (std::ostream& os, const std::vector<ValueT>& v)
{
    os << "[" << v.size() << "]";
    if (!v.empty()) {
        os << "(";
        katrin::KStringUtils::Join(os, v, ", ");
        os << ")";
    }
    return os;
}

template <class ValueT>
inline std::ostream& operator<< (std::ostream& os, const std::list<ValueT>& v)
{
    os << "[" << v.size() << "]";
    if (!v.empty()) {
        os << "(";
        katrin::KStringUtils::Join(os, v, ", ");
        os << ")";
    }
    return os;
}

template <class ValueT>
inline std::ostream& operator<< (std::ostream& os, const std::set<ValueT>& v)
{
    os << "[" << v.size() << "]";
    if (!v.empty()) {
        os << "(";
        katrin::KStringUtils::Join(os, v, ", ");
        os << ")";
    }
    return os;
}

template <class KeyT, class ValueT>
inline std::ostream& operator<< (std::ostream& os, const std::map<KeyT, ValueT>& v)
{
    os << "[" << v.size() << "]";
    if (!v.empty()) {
        os << "(";
        katrin::KStringUtils::Join(os, v);
        os << ")";
    }
    return os;
}

template <class InputT>
inline std::ostream& operator<< (std::ostream& os, const boost::numeric::ublas::matrix<InputT>& matrix)
{
    typedef typename boost::numeric::ublas::matrix<InputT>::size_type size_type;

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

template <class InputT>
inline std::ostream& operator<< (std::ostream& os, const std::vector<std::vector<InputT> >& matrix)
{
    typedef typename std::vector<std::vector<InputT> >::size_type size_type;

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

}

namespace boost {

template <class InputT, std::size_t S>
inline std::ostream& operator<< (std::ostream& os, const boost::array<InputT, S>& arr)
{
    os << "[" << arr.size() << "]";
    if (!arr.empty()) {
        os << "(";
        katrin::KStringUtils::Join(os, arr, ", ");
        os << ")";
    }
    return os;
}

}
#endif /* KSTRINGUTILS_H_ */
