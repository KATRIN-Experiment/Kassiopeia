#include "KSReadEventROOT.h"

namespace Kassiopeia
{

KSReadEventROOT::KSReadEventROOT(TFile* aFile) :
    KSReadIteratorROOT(aFile, (TTree*) (aFile->Get("EVENT_KEYS")), (TTree*) (aFile->Get("EVENT_DATA"))),
    fEventIndex(0),
    fFirstTrackIndex(0),
    fLastTrackIndex(0),
    fFirstStepIndex(0),
    fLastStepIndex(0)
{
    fData->SetBranchAddress("EVENT_INDEX", &fEventIndex);
    fData->SetBranchAddress("FIRST_TRACK_INDEX", &fFirstTrackIndex);
    fData->SetBranchAddress("LAST_TRACK_INDEX", &fLastTrackIndex);
    fData->SetBranchAddress("FIRST_STEP_INDEX", &fFirstStepIndex);
    fData->SetBranchAddress("LAST_STEP_INDEX", &fLastStepIndex);
}
KSReadEventROOT::~KSReadEventROOT() {}

unsigned int KSReadEventROOT::GetEventIndex() const
{
    return fEventIndex;
}
unsigned int KSReadEventROOT::GetFirstTrackIndex() const
{
    return fFirstTrackIndex;
}
unsigned int KSReadEventROOT::GetLastTrackIndex() const
{
    return fLastTrackIndex;
}
unsigned int KSReadEventROOT::GetFirstStepIndex() const
{
    return fFirstStepIndex;
}
unsigned int KSReadEventROOT::GetLastStepIndex() const
{
    return fLastStepIndex;
}

KSReadEventROOT& KSReadEventROOT::operator=(const unsigned int& aValue)
{
    *this << aValue;
    return *this;
}

}  // namespace Kassiopeia
