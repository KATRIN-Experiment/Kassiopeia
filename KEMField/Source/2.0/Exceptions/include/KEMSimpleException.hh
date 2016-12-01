/*
 * KEMSimpleException.hh
 *
 *  Created on: 16 Jun 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_EXCEPTIONS_INCLUDE_KEMSIMPLEEXCEPTION_HH_
#define KEMFIELD_SOURCE_2_0_EXCEPTIONS_INCLUDE_KEMSIMPLEEXCEPTION_HH_

#include <exception>
#include <string>

namespace KEMField{

class KEMSimpleException : public std::exception
{
public:
	KEMSimpleException(std::string information);
	~KEMSimpleException() _GLIBCXX_USE_NOEXCEPT;

	const char* what() const _GLIBCXX_USE_NOEXCEPT;

private:
	std::string fInformation;
};

}//KEMField



#endif /* KEMFIELD_SOURCE_2_0_EXCEPTIONS_INCLUDE_KEMSIMPLEEXCEPTION_HH_ */
