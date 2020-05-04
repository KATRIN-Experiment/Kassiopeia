/*
 * KKeyNotFoundException.cc
 *
 *  Created on: 19.05.2015
 *      Author: gosda
 */

#include "KKeyNotFoundException.hh"

using namespace std;

namespace KEMField
{

KKeyNotFoundException::KKeyNotFoundException(string container, string key, ErrorCode errorCode) :
    fContainer(container),
    fKey(key),
    fErrorCode(errorCode)
{}

KKeyNotFoundException::~KKeyNotFoundException() noexcept {}

const char* KKeyNotFoundException::what() const noexcept
{
    switch (fErrorCode) {
        case noEntry:
            return "The container contains no entry for the key.";
            break;
        case wrongType:
            return "The entry for the key is of an incompatible type.";
            break;
        default:
            return "Internal error of KKeyNotFoundException.";
    }
}

}  // namespace KEMField
