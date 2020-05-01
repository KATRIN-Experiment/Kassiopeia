#ifndef _Kassiopeia_KSReadStepROOT_h_
#define _Kassiopeia_KSReadStepROOT_h_

#include "KSReadIteratorROOT.h"

namespace Kassiopeia
{

class KSReadStepROOT : public KSReadIteratorROOT
{
  public:
    KSReadStepROOT(TFile* aFile);
    ~KSReadStepROOT() override;

  public:
    unsigned int GetStepIndex() const;

  public:
    KSReadStepROOT& operator=(const unsigned int& aValue);

  private:
    unsigned int fStepIndex;
};

}  // namespace Kassiopeia

#endif
