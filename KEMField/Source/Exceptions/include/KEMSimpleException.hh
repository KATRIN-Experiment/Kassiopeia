/*
 * KEMSimpleException.hh
 *
 *  Created on: 16 Jun 2015
 *      Author: wolfgang
 */

#ifndef KEMSIMPLEEXCEPTION_HH_
#define KEMSIMPLEEXCEPTION_HH_

#include <exception>
#include <string>

namespace KEMField
{

class KEMSimpleException : public std::exception
{
  public:
    KEMSimpleException(std::string information);
    ~KEMSimpleException() noexcept override;

    const char* what() const noexcept override;

  private:
    std::string fInformation;
};

}  // namespace KEMField


#endif /* KEMSIMPLEEXCEPTION_HH_ */
