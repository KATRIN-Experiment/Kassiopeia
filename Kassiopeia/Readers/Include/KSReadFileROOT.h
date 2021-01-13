#ifndef Kassiopeia_KSReadFileROOT_h_
#define Kassiopeia_KSReadFileROOT_h_

#include "KRootFile.h"
#include "KSReadEventROOT.h"
#include "KSReadFile.h"
#include "KSReadRunROOT.h"
#include "KSReadStepROOT.h"
#include "KSReadTrackROOT.h"

namespace Kassiopeia
{

class KSReadFileROOT : public KSReadFile
{
  public:
    using ObjectMap = std::map<std::string, KSReadObjectROOT*>;
    using ObjectIt = ObjectMap::iterator;
    using ObjectCIt = ObjectMap::const_iterator;
    using ObjectEntry = ObjectMap::value_type;

  public:
    KSReadFileROOT();
    ~KSReadFileROOT() override;

    bool TryFile(katrin::KRootFile* aFile);
    void OpenFile(katrin::KRootFile* aFile);
    void CloseFile();

    KSReadRunROOT& GetRun();
    KSReadEventROOT& GetEvent();
    KSReadTrackROOT& GetTrack();
    KSReadStepROOT& GetStep();

  protected:
    katrin::KRootFile* fRootFile;

    KSReadRunROOT* fRun;
    KSReadEventROOT* fEvent;
    KSReadTrackROOT* fTrack;
    KSReadStepROOT* fStep;
};

}  // namespace Kassiopeia

#endif
