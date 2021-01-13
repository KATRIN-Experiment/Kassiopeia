#include "KSReadRunROOT.h"

namespace Kassiopeia
{

KSReadRunROOT::KSReadRunROOT(TFile* aFile) :
    KSReadIteratorROOT(aFile, (TTree*) (aFile->Get("RUN_KEYS")), (TTree*) (aFile->Get("RUN_DATA"))),
    fRunIndex(0),
    fFirstEventIndex(0),
    fLastEventIndex(0),
    fFirstTrackIndex(0),
    fLastTrackIndex(0),
    fFirstStepIndex(0),
    fLastStepIndex(0)
{
    fData->SetBranchAddress("RUN_INDEX", &fRunIndex);
    fData->SetBranchAddress("FIRST_EVENT_INDEX", &fFirstEventIndex);
    fData->SetBranchAddress("LAST_EVENT_INDEX", &fLastEventIndex);
    fData->SetBranchAddress("FIRST_TRACK_INDEX", &fFirstTrackIndex);
    fData->SetBranchAddress("LAST_TRACK_INDEX", &fLastTrackIndex);
    fData->SetBranchAddress("FIRST_STEP_INDEX", &fFirstStepIndex);
    fData->SetBranchAddress("LAST_STEP_INDEX", &fLastStepIndex);
}
KSReadRunROOT::~KSReadRunROOT() = default;

unsigned int KSReadRunROOT::GetRunIndex() const
{
    return fRunIndex;
}
unsigned int KSReadRunROOT::GetLastRunIndex() const
{
    return (fData->GetEntries() - 1);
}
unsigned int KSReadRunROOT::GetFirstEventIndex() const
{
    return fFirstEventIndex;
}
unsigned int KSReadRunROOT::GetLastEventIndex() const
{
    return fLastEventIndex;
}
unsigned int KSReadRunROOT::GetFirstTrackIndex() const
{
    return fFirstTrackIndex;
}
unsigned int KSReadRunROOT::GetLastTrackIndex() const
{
    return fLastTrackIndex;
}
unsigned int KSReadRunROOT::GetFirstStepIndex() const
{
    return fFirstStepIndex;
}
unsigned int KSReadRunROOT::GetLastStepIndex() const
{
    return fLastStepIndex;
}

KSReadRunROOT& KSReadRunROOT::operator=(const unsigned int& aValue)
{
    *this << aValue;
    return *this;
}

}  // namespace Kassiopeia
