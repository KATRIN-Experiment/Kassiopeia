#ifndef KSATokenizer_HH__
#define KSATokenizer_HH__


#include "KSADefinitions.hh"

#include <string>
#include <vector>

namespace KEMField
{

/**
*
*@file KSATokenizer.hh
*@class KSATokenizer
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Dec 13 19:47:22 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KSATokenizer
{
  public:
    KSATokenizer()
    {
        fDelim = " ";  //default delim is space
        fString = nullptr;
        fIncludeEmptyTokens = false;
    };
    virtual ~KSATokenizer()
    {
        ;
    };

    void SetIncludeEmptyTokensTrue()
    {
        fIncludeEmptyTokens = true;
    };
    void SetIncludeEmptyTokensFalse()
    {
        fIncludeEmptyTokens = false;
    };

    void SetString(const std::string* aString)
    {
        fString = aString;
    };
    void SetDelimiter(const std::string& aDelim)
    {
        fDelim = aDelim;
    };
    void GetTokens(std::vector<std::string>* tokens) const
    {
        if (tokens != nullptr && fString != nullptr) {
            tokens->clear();
            if (fDelim.size() > 0) {

                size_t start = 0;
                size_t end = 0;
                size_t length = 0;
                while (end != std::string::npos) {
                    end = fString->find(fDelim, start);

                    if (end == std::string::npos) {
                        length = std::string::npos;
                    }
                    else {
                        length = end - start;
                    }


                    if (fIncludeEmptyTokens || ((length > 0) && (start < fString->size()))) {
                        tokens->push_back(fString->substr(start, length));
                    }

                    if (end > std::string::npos - fDelim.size()) {
                        start = std::string::npos;
                    }
                    else {
                        start = end + fDelim.size();
                    }
                }
            }
        }
    }


  protected:
    bool fIncludeEmptyTokens;
    std::string fDelim;
    const std::string* fString;
};

}  // namespace KEMField

#endif /* __KSATokenizer_H__ */
