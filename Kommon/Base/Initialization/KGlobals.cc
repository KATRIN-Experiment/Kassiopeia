/**
 * @file KGlobals.cc
 * @date Created on: 26.11.2021
 * @author Benedikt Bieringer <benedikt.b@wwu.de>
 */

#include "KGlobals.hh"
#include "KException.h"
 
using namespace katrin;

KGlobals::KGlobals() {fAccessed=false;}

KGlobals::~KGlobals() = default;

bool KGlobals::IsBatchMode()
{
    fAccessed = true;
    return fBatchMode;
}

void KGlobals::SetBatchMode(bool batchMode)
{
    if (fAccessed) {
        throw KException() << "KGlobals::SetBatchMode: Set after IsBatchMode was called!";
    }
    fBatchMode = batchMode;
}
