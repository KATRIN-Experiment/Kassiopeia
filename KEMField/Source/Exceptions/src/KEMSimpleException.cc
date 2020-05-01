/*
 * KEMSimpleException.cc
 *
 *  Created on: 16 Jun 2015
 *      Author: wolfgang
 */

#include "KEMSimpleException.hh"

using namespace std;

namespace KEMField
{

KEMSimpleException::KEMSimpleException(string information) : fInformation(information) {}

KEMSimpleException::~KEMSimpleException() noexcept {}

const char* KEMSimpleException::what() const noexcept
{
    return fInformation.c_str();
}

}  // namespace KEMField
