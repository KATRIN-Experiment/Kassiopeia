#include "KSReadFileROOT.h"

#include "KSReadersMessage.h"
#include "TFile.h"

using namespace katrin;

namespace Kassiopeia
{

KSReadFileROOT::KSReadFileROOT() : fRootFile(nullptr), fRun(nullptr), fEvent(nullptr), fTrack(nullptr), fStep(nullptr)
{}

KSReadFileROOT::~KSReadFileROOT() = default;

bool KSReadFileROOT::TryFile(KRootFile* aFile)
{
    if (aFile->Open(KFile::eRead) == true) {
        aFile->Close();
        return true;
    }
    return false;
}

void KSReadFileROOT::OpenFile(KRootFile* aFile)
{
    fRootFile = aFile;
    if (fRootFile->Open(KFile::eRead) == true) {

        fRun = new KSReadRunROOT(fRootFile->File());
        fEvent = new KSReadEventROOT(fRootFile->File());
        fTrack = new KSReadTrackROOT(fRootFile->File());
        fStep = new KSReadStepROOT(fRootFile->File());

        return;
    }

    readermsg(eError) << "cannot open file <" << fRootFile->GetName() << ">" << eom;
    return;
}
void KSReadFileROOT::CloseFile()
{
    if (fRootFile->Close() == true) {

        delete fRun;
        delete fEvent;
        delete fTrack;
        delete fStep;

        return;
    }

    readermsg(eError) << "cannot close file <" << fRootFile->GetName() << ">" << eom;
    return;
}

KSReadRunROOT& KSReadFileROOT::GetRun()
{
    return *fRun;
}
KSReadEventROOT& KSReadFileROOT::GetEvent()
{
    return *fEvent;
}
KSReadTrackROOT& KSReadFileROOT::GetTrack()
{
    return *fTrack;
}
KSReadStepROOT& KSReadFileROOT::GetStep()
{
    return *fStep;
}

}  // namespace Kassiopeia
