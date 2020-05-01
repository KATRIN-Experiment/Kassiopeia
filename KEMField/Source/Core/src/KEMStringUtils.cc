/*
 * KEMStringUtils.cc
 *
 *  Created on: 17 Aug 2015
 *      Author: wolfgang
 */

#include "KEMStringUtils.hh"

namespace KEMField
{

bool endsWith(std::string aString, std::string ending)
{
    if (aString.length() >= ending.length()) {
        return (0 == aString.compare(aString.length() - ending.length(), ending.length(), ending));
    }
    else {
        return false;
    }
}

} /* namespace KEMField */
