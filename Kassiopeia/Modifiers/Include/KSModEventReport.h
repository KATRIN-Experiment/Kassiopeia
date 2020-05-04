#ifndef KSCKassiopeia_KSModEventReport_h_
#define KSCKassiopeia_KSModEventReport_h_

#include "KSComponentTemplate.h"
#include "KSEventModifier.h"

namespace Kassiopeia
{
class KSEvent;

//This is a super simple example of an event modifier that does
//essentially nothing (it can be used as a static event modifier too).
//If a user wishes to write their own event modifier they should place it
//in a separate library (not in Kassiopeia/Modifiers) or they will introduce
//a cyclic dependency between the KassiopieaModifiers and KassiopeiaSimulation libraries.
//This is is also true for run/track/step modifiers as well.

class KSModEventReport : public KSComponentTemplate<KSModEventReport, KSEventModifier>
{
  public:
    KSModEventReport();
    KSModEventReport(const KSModEventReport& /*aCopy*/);
    KSModEventReport* Clone() const override;
    ~KSModEventReport() override;

    bool ExecutePreEventModification(KSEvent& /*anEvent*/) override;
    bool ExecutePostEventModification(KSEvent& /*anEvent*/) override;

  private:
};

}  // namespace Kassiopeia

#endif
