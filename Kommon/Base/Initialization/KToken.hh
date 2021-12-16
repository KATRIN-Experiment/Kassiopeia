#ifndef Kommon_KToken_hh_
#define Kommon_KToken_hh_

#include "KInitializationMessage.hh"
#include "KBaseStringUtils.h"
#include "KException.h"
#include "KLogger.h"

#include <sstream>
#include <string>
#include <limits.h>

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
    try {
        return KBaseStringUtils::Convert<XDataType>(fValue);
    } catch (KException &err) {
        throw KException() << "Unable to process path <" << fPath << "> in file <" << fFile 
                           << "> at line <" << fLine << "> at column <" << fColumn << ">: "
                           << err.what();
    }
}

/* KToken string conversion specializations */

template<class OutputT> inline const std::vector<OutputT> KToken::AsVector() const
{
    return KBaseStringUtils::SplitTrimAndConvert<OutputT>(fValue, ";, |/");
}

template<class OutputT> inline const std::pair<OutputT, OutputT> KToken::AsPair() const
{
    std::vector<OutputT> tmp = AsVector<OutputT>();
    size_t n = tmp.size();
    if (n != 2) {
        // Throwing exception as there is no possible way for a scientifically reasonable recovery when processing fails
        throw KException() << "error processing token <" << KBaseStringUtils::EscapeMostly(fValue)
                          << "> as a pair of double values, replaced with zeros." << ret
                          << "in path <" << fPath << "> in file <" << fFile << "> at line <" << fLine << "> at column <"
                          << fColumn << ">" << eom;
    }
    return std::make_pair(tmp[0], tmp[1]);
}

template<> inline unsigned int KToken::GetValue<unsigned int>() const
{
    try {
        return KBaseStringUtils::Convert<unsigned int>(fValue);
    } catch (KException &err) {
        if(fValue == "-1") {
            KLOGGER("kommon.ktoken");
            
            KWARN("While processing path <" << fPath << "> in file <" << fFile 
            << "> at line <" << fLine << "> at column <" << fColumn << ">: "
            << "<-1> is interpreted as <unsigned int>, resulting in " << (UINT_MAX - 1));
            return UINT_MAX - 1;
                // Needed for number_of_bifurcations value
                // It is not clear whether such a large bifurcation value actually makes sense,
                // see https://nuserv.uni-muenster.de:8443/katrin-git/kasper/-/merge_requests/605#note_9609 .
                // But since it is used very frequently, this was added again for compatibility with
                // current configuration files.
        }
        throw KException() << "Unable to process path <" << fPath << "> in file <" << fFile 
                           << "> at line <" << fLine << "> at column <" << fColumn << ">: "
                           << err.what();
    }
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

}  // namespace katrin

#endif
