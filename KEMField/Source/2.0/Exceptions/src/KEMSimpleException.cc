/*
 * KEMSimpleException.cc
 *
 *  Created on: 16 Jun 2015
 *      Author: wolfgang
 */

#include "KEMSimpleException.hh"

using namespace std;

namespace KEMField {

KEMSimpleException::KEMSimpleException(
		string information) :
				fInformation(information)
{
}

KEMSimpleException::~KEMSimpleException() _GLIBCXX_USE_NOEXCEPT
{
}

const char* KEMSimpleException::what() const _GLIBCXX_USE_NOEXCEPT
{
	return fInformation.c_str();
}

}//KEMField



