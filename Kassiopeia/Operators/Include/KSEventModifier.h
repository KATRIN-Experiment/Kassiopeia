#ifndef Kassiopeia_KSEventModifier_h_
#define Kassiopeia_KSEventModifier_h_

#include "KSComponentTemplate.h"

namespace Kassiopeia
{
class KSEvent;

class KSEventModifier : public KSComponentTemplate<KSEventModifier>
{
  public:
    KSEventModifier();
    ~KSEventModifier() override;

  public:
    //returns true if any of the state variables of anEvent are changed
    virtual bool ExecutePreEventModification(KSEvent& anEvent) = 0;

    //returns true if any of the state variables of anEvent are changed
    virtual bool ExecutePostEventModification(KSEvent& anEvent) = 0;
};

}  // namespace Kassiopeia

#endif
