/*
 * KKeyNotFoundException.cc
 *
 *  Created on: 19.05.2015
 *      Author: gosda
 */

#include "KKeyNotFoundException.hh"

#include <utility>

using namespace std;

namespace KEMField
{

KKeyNotFoundException::KKeyNotFoundException(const string& container, const string& key, ErrorCode errorCode) :
    fContainer(std::move(container)),
    fKey(std::move(key)),
    fErrorCode(errorCode)
{}

KKeyNotFoundException::~KKeyNotFoundException() noexcept = default;

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
