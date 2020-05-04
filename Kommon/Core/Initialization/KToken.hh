#ifndef Kommon_KToken_hh_
#define Kommon_KToken_hh_

#include "KInitializationMessage.hh"
#include "KStringUtils.h"

#include <sstream>
#include <string>

namespace katrin
{

class KToken
{
  public:
    KToken();
    KToken(const KToken& aToken);
    virtual ~KToken();

    virtual KToken* Clone() = 0;

  public:
    template<class OutputT> inline const std::vector<OutputT> AsVector() const;

    template<class OutputT> inline const std::pair<OutputT, OutputT> AsPair() const;

    void SetValue(const std::string& aValue);
    const std::string& GetValue() const;

    template<typename XDataType> XDataType GetValue() const;

    void SetPath(const std::string& aPath);
    const std::string& GetPath() const;

    void SetFile(const std::string& aFile);
    const std::string& GetFile() const;

    void SetLine(const int& aLine);
    const int& GetLine() const;

    void SetColumn(const int& aColumn);
    const int& GetColumn() const;

  private:
    std::string fValue;

    std::string fPath;
    std::string fFile;
    int fLine;
    int fColumn;
};

template<typename XDataType> inline XDataType KToken::GetValue() const
{
    std::istringstream Converter(fValue);
    XDataType Data;
    Converter >> Data;
    if (Converter.fail() || (Data != Data))  // also check for NaN
    {
        std::string TypeName = KMessage::TypeName<XDataType>();
        initmsg(eWarning) << "error processing token <" << fValue << "> with type <" << TypeName << ">, replaced with <"
                          << Data << ">" << ret;
        initmsg(eWarning) << "in path <" << fPath << "> in file <" << fFile << "> at line <" << fLine << "> at column <"
                          << fColumn << ">" << eom;
    }
    return Data;
}

template<> inline bool KToken::GetValue<bool>() const
{
    if (fValue == std::string("0") || fValue == std::string("") || fValue == std::string("null") ||
        fValue == std::string("Null") || fValue == std::string("NULL") || fValue == std::string("nan") ||
        fValue == std::string("NaN") || fValue == std::string("NAN") || fValue == std::string("none") ||
        fValue == std::string("None") || fValue == std::string("None") || fValue == std::string("false") ||
        fValue == std::string("False") || fValue == std::string("FALSE") || fValue == std::string("no") ||
        fValue == std::string("No") || fValue == std::string("NO")) {
        return false;
    }
    return true;
}

/* KToken string conversion specializations */

template<class OutputT> inline const std::vector<OutputT> KToken::AsVector() const
{
    return KStringUtils::Split<OutputT>(fValue, ";, |/");
}

template<class OutputT> inline const std::pair<OutputT, OutputT> KToken::AsPair() const
{
    std::vector<OutputT> tmp;
    size_t n = KStringUtils::Split<OutputT>(fValue, ";, |/", tmp);
    if (n != 2) {
        initmsg(eWarning) << "error processing token <" << fValue
                          << "> as a pair of double values, replaced with zeros." << ret;
        initmsg(eWarning) << "in path <" << fPath << "> in file <" << fFile << "> at line <" << fLine << "> at column <"
                          << fColumn << ">" << eom;
        return std::make_pair(0, 0);
    }
    return std::make_pair(tmp[0], tmp[1]);
}

template<> inline int8_t KToken::GetValue<int8_t>() const
{
    const auto helper = KToken::GetValue<int>();
    return (helper >= -128 && helper <= 127) ? helper : 0;
}

template<> inline uint8_t KToken::GetValue<uint8_t>() const
{
    const auto helper = KToken::GetValue<int>();
    return (helper >= 0 && helper <= 255) ? helper : 0;
}

template<> inline std::vector<double> KToken::GetValue<std::vector<double>>() const
{
    return AsVector<double>();
}

template<> inline std::vector<int> KToken::GetValue<std::vector<int>>() const
{
    return AsVector<int>();
}

template<> inline std::vector<unsigned int> KToken::GetValue<std::vector<unsigned int>>() const
{
    return AsVector<unsigned int>();
}

template<> inline std::pair<double, double> KToken::GetValue<std::pair<double, double>>() const
{
    return AsPair<double>();
}

template<> inline std::pair<int, int> KToken::GetValue<std::pair<int, int>>() const
{
    return AsPair<int>();
}

template<> inline std::pair<unsigned int, unsigned int> KToken::GetValue<std::pair<unsigned int, unsigned int>>() const
{
    return AsPair<unsigned int>();
}

template<> inline std::string KToken::GetValue<std::string>() const
{
    return fValue;
}

}  // namespace katrin

#endif
