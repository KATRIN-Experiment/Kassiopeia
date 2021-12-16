/**
 * @file KBaseStringUtils.h
 *
 * @date Created on: 26.11.2021
 * @author Benedikt Bieringer <benedikt.b@wwu.de>
 */

#ifndef KBASESTRINGUTILS_H_
#define KBASESTRINGUTILS_H_

#include <type_traits>
#include <vector>
#include <sstream>
#include <string>
#include <cctype>
#include <algorithm>
#include <functional>

#include <cxxabi.h>  // needed to convert typename to string

// For operator<< collection
#include <utility>
#include <array>
#include <deque>
#include <forward_list>
#include <list>
#include <stack>
#include <queue>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>

#include "KException.h"

namespace katrin
{

/**
 * Contains static functions for joining, exploding and serializing string sequences.
 */
class KBaseStringUtils
{
public:
    // New functions
    template<typename XDataType> static XDataType Convert(std::string str);
    template<class OutputT> static std::vector<OutputT> SplitAndConvert(std::string valueCopy, std::string delimiters);

    static std::string Replace(const std::string value, const std::string from, const std::string to);
    static std::string EscapeMostly(const std::string value);

    // Migrated from KStringUtils.h

    /**
     * Helper function to convert typename to human-readable string, see: http://stackoverflow.com/a/19123821
     */
    template<typename XDataType> static std::string TypeName();

    static std::string Trim(const std::string input);
    static std::string TrimLeft(const std::string input);
    static std::string TrimRight(const std::string input);

    static bool Equals(const std::string, const std::string);
    static bool IEquals(const std::string, const std::string);

    template<class SequenceT, class SeparatorT>
    static std::ostream& Join(std::ostream& stream, const SequenceT& sequence, const SeparatorT& separator,
                              std::function<void (std::ostream&, const typename SequenceT::const_iterator&)> formatter
                              = [](std::ostream& os, const typename SequenceT::const_iterator&it) { os << *it; });

    template<class SequenceT, class SeparatorT>
    static std::string Join(const SequenceT& sequence, const SeparatorT& separator, int precision = -1);

    template<class KeyT, class ValueT, class SeparatorT>
    static std::ostream& Join(std::ostream& stream, const std::map<KeyT, ValueT>& map, const SeparatorT& separator);

    template<class KeyT, class ValueT, class SeparatorT>
    static std::string Join(const std::map<KeyT, ValueT>& map, const SeparatorT& separator, int precision = -1);
};

template<typename XDataType> XDataType KBaseStringUtils::Convert(std::string str)
{
    std::istringstream Converter(str);
    XDataType Data;
    Converter >> Data;
    std::string rest;
    if (Converter.fail() || (Data != Data))  // also check for NaN
    {
        // Throwing exception as there is no possible way for a scientifically reasonable recovery when conversion fails
        throw KException() << "Error processing <" << EscapeMostly(str) << "> with type <" << TypeName<XDataType>() << ">.";

    }
    if (Converter.rdbuf()->in_avail() > 0) {
        rest = Converter.str().substr(Converter.tellg());
    }
    if (Converter.fail() || !(std::all_of(rest.begin(),rest.end(),::isspace)))
    {
        throw KException() << "Error processing <" << EscapeMostly(str) << "> with type <" << TypeName<XDataType>() << ">: "
                           << "Uninterpreted part remaining: '" << EscapeMostly(rest) << "'. "
                           << "This error message was added in November 2021. Before, the remaining part was silently ignored.";
    }
    if (std::is_unsigned<XDataType>::value && str.find("-") != std::string::npos) {
        throw KException() << "Error processing <" << EscapeMostly(str)  << "> with unsigned type <" << TypeName<XDataType>() << ">. "
                           << "This error message was added in November 2021. Before, if larger than -4294967296, "
                           << "the value was silently casted to be positive by adding 4294967296.";
    }
    return Data;
}

template<> inline bool KBaseStringUtils::Convert<bool>(std::string str)
{
    if (str == std::string("0") || str == std::string("") || str == std::string("null") ||
        str == std::string("Null") || str == std::string("NULL") || str == std::string("nan") ||
        str == std::string("NaN") || str == std::string("NAN") || str == std::string("none") ||
        str == std::string("None") || str == std::string("None") || str == std::string("false") ||
        str == std::string("False") || str == std::string("FALSE") || str == std::string("no") ||
        str == std::string("No") || str == std::string("NO")) {
        return false;
    }
    return true;
}

template<> inline int8_t KBaseStringUtils::Convert<int8_t>(std::string str)
{
    const auto helper = KBaseStringUtils::Convert<int>(str);
    if (helper < -128 || helper > 127)
    {
        throw KException() << "Error processing <" << EscapeMostly(str) << "> with type <int8_t>: Out of bounds: '" << helper << "'. "
                           << "This error message was added in November 2021. Before, this value was silently replaced by '0'.";
    }
    return helper;
}

template<> inline uint8_t KBaseStringUtils::Convert<uint8_t>(std::string str)
{
    const auto helper = KBaseStringUtils::Convert<int>(str);
    if (helper < 0 || helper > 255)
    {
        throw KException() << "Error processing token <" << EscapeMostly(str) << "> with type <uint8_t>: Out of bounds: '" << helper << "'. "
                           << "This error message was added in November 2021. Before, this value was silently replaced by '0'.";
    }
    return helper;
}

template<> inline std::string KBaseStringUtils::Convert<std::string>(std::string str)
{
    return str;
}

template<class OutputT> std::vector<OutputT> KBaseStringUtils::SplitAndConvert(std::string valueCopy, std::string delimiters)
{
    std::vector<OutputT> res;
    std::string tmp;

    try {
        // Split
        for (char& c : valueCopy) {
            if (delimiters.find(c) == std::string::npos) {
                tmp += c;
                continue;
            }

            if (tmp.size() != 0) {
                // Convert
                res.push_back(Convert<OutputT>(tmp));
                tmp = "";
            }
        }

        if (tmp.size() != 0) {
            // Convert last entry
            res.push_back(Convert<OutputT>(tmp));
        }
    } catch (KException &err) {
        throw KException() << "Unable to process token <" << EscapeMostly(valueCopy) << ">: "
                            << err.what();
    }

    return res;
}

// Migrated following functions from KStringUtils.h to KBaseStringUtils.h

template<class SequenceT, class SeparatorT>
inline std::ostream& KBaseStringUtils::Join(std::ostream& stream, const SequenceT& sequence, const SeparatorT& separator,
                                        std::function<void (std::ostream&, const typename SequenceT::const_iterator&)> formatter)
{
    // Parse input
    auto itBegin = std::begin(sequence);
    auto itEnd = std::end(sequence);

    // Append first element
    if (itBegin != itEnd) {
        formatter(stream, itBegin);
        ++itBegin;
    }
    for (; itBegin != itEnd; ++itBegin) {
        stream << separator; // This was formerly boost::as_literal(separator) for no apparent reason
        formatter(stream, itBegin);
    }
    return stream;
}

template<class SequenceT, class SeparatorT>
inline std::string KBaseStringUtils::Join(const SequenceT& sequence, const SeparatorT& separator, int precision)
{
    std::ostringstream result;
    if (precision >= 0)
        result.precision(precision);
    Join(result, sequence, separator);
    return result.str();
}

template<class KeyT, class ValueT, class SeparatorT>
inline std::ostream& KBaseStringUtils::Join(std::ostream& stream, const std::map<KeyT, ValueT>& map, const SeparatorT& separator)
{
    Join(stream, map, separator, [&](auto& os, auto& it) { os << "[" << it->first << "] " << it->second; });
    return stream;
}

template<class KeyT, class ValueT, class SeparatorT>
inline std::string KBaseStringUtils::Join(const std::map<KeyT, ValueT>& map, const SeparatorT& separator, int precision)
{
    std::ostringstream result;
    if (precision >= 0)
        result.precision(precision);
    Join(result, map, separator);
    return result.str();
}

template<typename XDataType> std::string KBaseStringUtils::TypeName()
{
    int StatusFlag;
    std::string TypeName = typeid(XDataType).name();
    char* DemangledName = abi::__cxa_demangle(TypeName.c_str(), nullptr, nullptr, &StatusFlag);
    if (StatusFlag == 0) {
        TypeName = std::string(DemangledName);
        free(DemangledName);
    }
    return TypeName;
}

}  // namespace katrin


// 2021-11-26: Migrated operator<< collection from KStringUtils.h to KBaseStringUtils.h
namespace std {

template<class Value1T, class Value2T> inline std::ostream& operator<<(std::ostream& os, const std::pair<Value1T,Value2T>& v)
{
    os << "[" << 2 << "]";
    os << "(";
    os << v.first << ", " << v.second;
    os << ")";
    return os;
}

template<class InputT, std::size_t S> inline std::ostream& operator<<(std::ostream& os, const std::array<InputT, S>& v)
{
    os << "[" << v.size() << "]";
    if (!v.empty()) {
        os << "(";
        katrin::KBaseStringUtils::Join(os, v, ", ");
        os << ")";
    }
    return os;
}

template<class ValueT> inline std::ostream& operator<<(std::ostream& os, const std::vector<ValueT>& v)
{
    os << "[" << v.size() << "]";

    if (!v.empty()) {
        os << "(";
        katrin::KBaseStringUtils::Join(os, v, ", ");
        os << ")";
    }
    return os;
}

template<class ValueT> inline std::ostream& operator<<(std::ostream& os, const std::deque<ValueT>& v)
{
    os << "[" << v.size() << "]";

    if (!v.empty()) {
        os << "(";
        katrin::KBaseStringUtils::Join(os, v, ", ");
        os << ")";
    }
    return os;
}

template<class ValueT> inline std::ostream& operator<<(std::ostream& os, const std::forward_list<ValueT>& v)
{
    os << "[" << v.size() << "]";
    if (!v.empty()) {
        os << "(";
        katrin::KBaseStringUtils::Join(os, v, ", ");
        os << ")";
    }
    return os;
}

template<class ValueT> inline std::ostream& operator<<(std::ostream& os, const std::list<ValueT>& v)
{
    os << "[" << v.size() << "]";

    if (!v.empty()) {
        os << "(";
        katrin::KBaseStringUtils::Join(os, v, ", ");
        os << ")";
    }
    return os;
}

template<class ValueT> inline std::ostream& operator<<(std::ostream& os, const std::stack<ValueT>& v)
{
    os << "[" << v.size() << "]";
    if (!v.empty()) {
        os << "(";
        katrin::KBaseStringUtils::Join(os, v, ", ");
        os << ")";
    }
    return os;
}

template<class ValueT> inline std::ostream& operator<<(std::ostream& os, const std::queue<ValueT>& v)
{
    os << "[" << v.size() << "]";
    if (!v.empty()) {
        os << "(";
        katrin::KBaseStringUtils::Join(os, v, ", ");
        os << ")";
    }
    return os;
}

template<class ValueT> inline std::ostream& operator<<(std::ostream& os, const std::priority_queue<ValueT>& v)
{
    os << "[" << v.size() << "]";
    if (!v.empty()) {
        os << "(";
        katrin::KBaseStringUtils::Join(os, v, ", ");
        os << ")";
    }
    return os;
}

template<class ValueT> inline std::ostream& operator<<(std::ostream& os, const std::set<ValueT>& v)
{
    os << "[" << v.size() << "]";

    if (!v.empty()) {
        os << "(";
        katrin::KBaseStringUtils::Join(os, v, ", ");
        os << ")";
    }
    return os;
}

template<class ValueT> inline std::ostream& operator<<(std::ostream& os, const std::multiset<ValueT>& v)
{
    os << "[" << v.size() << "]";
    if (!v.empty()) {
        os << "(";
        katrin::KBaseStringUtils::Join(os, v, ", ");
        os << ")";
    }
    return os;
}

template<class ValueT> inline std::ostream& operator<<(std::ostream& os, const std::unordered_set<ValueT>& v)
{
    os << "[" << v.size() << "]";
    if (!v.empty()) {
        os << "(";
        katrin::KBaseStringUtils::Join(os, v, ", ", [&](auto& os, auto& it) { os << "[" << it->first << "] " << it->second; });
        os << ")";
    }
    return os;
}

template<class KeyT, class ValueT> inline std::ostream& operator<<(std::ostream& os, const std::map<KeyT, ValueT>& v)
{
    os << "[" << v.size() << "]";

    if (!v.empty()) {
        os << "(";
        katrin::KBaseStringUtils::Join(os, v, ", ");
        os << ")";
    }
    return os;
}

template<class KeyT, class ValueT> inline std::ostream& operator<<(std::ostream& os, const std::multimap<KeyT, ValueT>& v)
{
    os << "[" << v.size() << "]";
    if (!v.empty()) {
        os << "(";
        katrin::KBaseStringUtils::Join(os, v, ", ", [&](auto& os, auto& it) { os << "[" << it->first << "] " << it->second; });
        os << ")";
    }
    return os;
}

template<class KeyT, class ValueT> inline std::ostream& operator<<(std::ostream& os, const std::unordered_map<KeyT, ValueT>& v)
{
    os << "[" << v.size() << "]";
    if (!v.empty()) {
        os << "(";
        katrin::KBaseStringUtils::Join(os, v, ", ", [&](auto& os, auto& it) { os << "[" << it->first << "] " << it->second; });
        os << ")";
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

}  // namespace std

#endif /* KBASESTRINGUTILS_H_ */
