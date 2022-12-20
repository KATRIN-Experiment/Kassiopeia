/**
 * @file KGlobals.cc
 * @date Created on: 26.11.2021
 * @author Benedikt Bieringer <benedikt.b@wwu.de>
 */

#include "KGlobals.hh"
//#include "KException.h"
#include "KInitializationMessage.hh"

using namespace katrin;

KGlobals::KGlobals() : fAccessed(false), fBatchMode(false), fVerbosityLevel(0) {}

KGlobals::~KGlobals() = default;

bool KGlobals::IsBatchMode()
{
    fAccessed = true;
    return fBatchMode;
}

int KGlobals::VerbosityLevel()
{
    fAccessed = true;
    return fVerbosityLevel;
}

void KGlobals::SetBatchMode(bool batchMode)
{
    if (fAccessed) {
        //throw KException() << "KGlobals::SetBatchMode: Set after IsBatchMode was called!";
        initmsg(eWarning) << "KGlobals::SetBatchMode: Set after IsBatchMode was called!" << eom;
    }
    fBatchMode = batchMode;
}

void KGlobals::SetVerbosityLevel(int verbosityLevel)
{
    if (fAccessed) {
        //throw KException() << "KGlobals::SetVerbosityLevel: Set after VerbosityLevel was called!";
        initmsg(eWarning) << "KGlobals::SetVerbosityLevel: Set after VerbosityLevel was called!" << eom;
    }
    fVerbosityLevel = verbosityLevel;
}
