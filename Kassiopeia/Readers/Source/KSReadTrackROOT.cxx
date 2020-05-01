#include "KSReadTrackROOT.h"

namespace Kassiopeia
{

KSReadTrackROOT::KSReadTrackROOT(TFile* aFile) :
    KSReadIteratorROOT(aFile, (TTree*) (aFile->Get("TRACK_KEYS")), (TTree*) (aFile->Get("TRACK_DATA"))),
    fTrackIndex(0),
    fFirstStepIndex(0),
    fLastStepIndex(0)
{
    fData->SetBranchAddress("TRACK_INDEX", &fTrackIndex);
    fData->SetBranchAddress("FIRST_STEP_INDEX", &fFirstStepIndex);
    fData->SetBranchAddress("LAST_STEP_INDEX", &fLastStepIndex);
}
KSReadTrackROOT::~KSReadTrackROOT() {}

unsigned int KSReadTrackROOT::GetTrackIndex() const
{
    return fTrackIndex;
}
unsigned int KSReadTrackROOT::GetFirstStepIndex() const
{
    return fFirstStepIndex;
}
unsigned int KSReadTrackROOT::GetLastStepIndex() const
{
    return fLastStepIndex;
}

KSReadTrackROOT& KSReadTrackROOT::operator=(const unsigned int& aValue)
{
    *this << aValue;
    return *this;
}

}  // namespace Kassiopeia
