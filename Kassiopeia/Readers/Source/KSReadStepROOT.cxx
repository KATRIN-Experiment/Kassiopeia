#include "KSReadStepROOT.h"

namespace Kassiopeia
{

KSReadStepROOT::KSReadStepROOT(TFile* aFile) :
    KSReadIteratorROOT(aFile, (TTree*) (aFile->Get("STEP_KEYS")), (TTree*) (aFile->Get("STEP_DATA"))),
    fStepIndex(0)
{
    fData->SetBranchAddress("STEP_INDEX", &fStepIndex);
}
KSReadStepROOT::~KSReadStepROOT() {}

unsigned int KSReadStepROOT::GetStepIndex() const
{
    return fStepIndex;
}

KSReadStepROOT& KSReadStepROOT::operator=(const unsigned int& aValue)
{
    *this << aValue;
    return *this;
}

}  // namespace Kassiopeia
