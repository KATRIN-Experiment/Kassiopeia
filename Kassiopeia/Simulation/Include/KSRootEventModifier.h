#ifndef Kassiopeia_KSRootEventModifier_h_
#define Kassiopeia_KSRootEventModifier_h_

#include "KSEvent.h"
#include "KSEventModifier.h"
#include "KSList.h"
#include "KSParticle.h"
#include "KSStep.h"
#include "KSTrack.h"

namespace Kassiopeia
{

class KSRootEventModifier : public KSComponentTemplate<KSRootEventModifier, KSEventModifier>
{
  public:
    KSRootEventModifier();
    KSRootEventModifier(const KSRootEventModifier& aCopy);
    KSRootEventModifier* Clone() const override;
    ~KSRootEventModifier() override;

    //**********
    // modifier
    //**********

  protected:
    bool ExecutePreEventModification(KSEvent& anEvent) override;
    bool ExecutePostEventModification(KSEvent& anEvent) override;

    //***********
    //composition
    //***********

  public:
    void AddModifier(KSEventModifier* aModifier);
    void RemoveModifier(KSEventModifier* aModifier);

  private:
    KSList<KSEventModifier> fModifiers;
    KSEventModifier* fModifier;

    //******
    //action
    //******

  public:
    void SetEvent(KSEvent* anEvent);

    bool ExecutePreEventModification();
    bool ExecutePostEventModification();

    void PushUpdateComponent() override;
    void PushDeupdateComponent() override;

  private:
    KSEvent* fEvent;
};


}  // namespace Kassiopeia

#endif
