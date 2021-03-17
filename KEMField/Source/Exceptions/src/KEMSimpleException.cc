/*
 * KEMSimpleException.cc
 *
 *  Created on: 16 Jun 2015
 *      Author: wolfgang
 */

#include "KEMSimpleException.hh"

#include <utility>

using namespace std;

namespace KEMField
{

KEMSimpleException::KEMSimpleException(const string& information) : fInformation(std::move(information)) {}

KEMSimpleException::~KEMSimpleException() noexcept = default;

const char* KEMSimpleException::what() const noexcept
{
    return fInformation.c_str();
}

}  // namespace KEMField
