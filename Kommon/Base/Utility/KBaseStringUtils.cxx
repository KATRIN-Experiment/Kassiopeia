#include "KBaseStringUtils.h"

namespace katrin
{

bool KBaseStringUtils::Equals(const std::string r1, const std::string r2)
{
    return ( (r1.size() == r2.size() ) &&
        std::equal(r1.begin(), r1.end(), r2.begin()) );
}

bool KBaseStringUtils::IEquals(const std::string r1, const std::string r2)
{
    return ( (r1.size() == r2.size() ) &&
        std::equal(r1.begin(), r1.end(), r2.begin(), [](unsigned char c1, unsigned char c2) {
            return (c1 == c2) || (std::toupper(c1) == std::toupper(c2));
    }) );
}

std::string KBaseStringUtils::Trim(const std::string input)
{
    return TrimLeft(TrimRight(input));
}

std::string KBaseStringUtils::TrimLeft(const std::string input)
{
    std::string output(input);
    output.erase(output.begin(), std::find_if(output.begin(), output.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    return output;
}

std::string KBaseStringUtils::TrimRight(const std::string input)
{
    std::string output(input);
    output.erase(std::find_if(output.rbegin(), output.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), output.end());
    return output;
}

std::string KBaseStringUtils::Replace(const std::string value, const std::string from, const std::string to)
{
        std::string result(value);
        auto pos = result.find(from);
        auto from_len = from.length();
        while(pos != std::string::npos)
        {
            result.replace(pos, from_len, to);
            pos = result.find(from);
        }
        return result;
}

/**
 * Escape a string.
 * Ignores numeric escape sequences, conditional escape sequences
 * and universal character names.
 */
std::string KBaseStringUtils::EscapeMostly(const std::string value)
{
    std::vector<std::pair<std::string,std::string>> sequences = {
        {"\'", "\\'"},
        {"\"", "\\"},
        {"\?", "\\\?"},
        {"\\", "\\\\"},
        {"\a", "\\a"},
        {"\b", "\\b"},
        {"\f", "\\f"},
        {"\n", "\\n"},
        {"\r", "\\r"},
        {"\t", "\\t"},
        {"\v", "\\v"},
    };
    std::string result(value);
    for(auto& pair : sequences)
    {
        result = Replace(result, pair.first, pair.second);
    }
    return result;
            
        ;
}

}  // namespace katrin
