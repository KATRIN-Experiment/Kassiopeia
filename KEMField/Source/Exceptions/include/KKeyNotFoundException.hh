/*
 * KKeyNotFoundException.hh
 *	An exception for the KEMToolbox to throw if no corresponding entry is found.
 *
 *  Created on: 19.05.2015
 *      Author: gosda
 */

#ifndef KKEYNOTFOUNDEXCEPTION_HH_
#define KKEYNOTFOUNDEXCEPTION_HH_

#include <KException.h>
#include <string>

namespace KEMField
{

class KKeyNotFoundException : public katrin::KException
{
  public:
    enum ErrorCode
    {
        noEntry,
        wrongType
    };

    KKeyNotFoundException(const std::string& container, const std::string& key, ErrorCode errorCode);
    ~KKeyNotFoundException() noexcept override;

    const char* what() const noexcept override;

  private:
    std::string fContainer;
    std::string fKey;
    ErrorCode fErrorCode;
};

}  // namespace KEMField


#endif /* KKEYNOTFOUNDEXCEPTION_HH_ */
