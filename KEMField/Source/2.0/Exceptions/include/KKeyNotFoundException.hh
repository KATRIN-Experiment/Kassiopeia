/*
 * KKeyNotFoundException.hh
 *	An exception for the KEMToolbox to throw if no corresponding entry is found.
 *
 *  Created on: 19.05.2015
 *      Author: gosda
 */

#ifndef KKEYNOTFOUNDEXCEPTION_HH_
#define KKEYNOTFOUNDEXCEPTION_HH_

#include <exception>
#include <string>

namespace KEMField{

class KKeyNotFoundException : public std::exception
{
public:
	enum ErrorCode {noEntry, wrongType};

	KKeyNotFoundException(std::string container, std::string key, ErrorCode errorCode);
	~KKeyNotFoundException() noexcept;

	const char* what() const noexcept;

private:
	std::string fContainer;
	std::string fKey;
	ErrorCode fErrorCode;

};

}//KEMField



#endif /* KKEYNOTFOUNDEXCEPTION_HH_ */
