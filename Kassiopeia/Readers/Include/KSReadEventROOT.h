#ifndef _Kassiopeia_KSReadEventROOT_h_
#define _Kassiopeia_KSReadEventROOT_h_

#include "KSReadIteratorROOT.h"

namespace Kassiopeia
{

class KSReadEventROOT : public KSReadIteratorROOT
{
  public:
    KSReadEventROOT(TFile* aFile);
    ~KSReadEventROOT() override;

  public:
    unsigned int GetEventIndex() const;
    unsigned int GetFirstTrackIndex() const;
    unsigned int GetLastTrackIndex() const;
    unsigned int GetFirstStepIndex() const;
    unsigned int GetLastStepIndex() const;

  public:
    KSReadEventROOT& operator=(const unsigned int& aValue);

  private:
    unsigned int fEventIndex;
    unsigned int fFirstTrackIndex;
    unsigned int fLastTrackIndex;
    unsigned int fFirstStepIndex;
    unsigned int fLastStepIndex;
};

}  // namespace Kassiopeia

#endif
