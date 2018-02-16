/*
 * KKeyNotFoundException.cc
 *
 *  Created on: 19.05.2015
 *      Author: gosda
 */

#include "KKeyNotFoundException.hh"

using namespace std;

namespace KEMField {

KKeyNotFoundException::KKeyNotFoundException(
		string container,
		string key,
		ErrorCode errorCode) :
				fContainer(container),fKey(key),fErrorCode(errorCode)
{
}

KKeyNotFoundException::~KKeyNotFoundException() noexcept
{
}

const char* KKeyNotFoundException::what() const noexcept
{
	switch(fErrorCode) {
	case noEntry:
		return(fContainer + " contains no entry for key \"" + fKey + "\".").c_str();
		break;
	case wrongType:
		return("The entry in " +fContainer + " for the key \"" + fKey + "\" is "
				"of an incompatible type.").c_str();
		break;
	default:
		return "Internal error of KKeyNotFoundException.";
	}
}

}//KEMField


