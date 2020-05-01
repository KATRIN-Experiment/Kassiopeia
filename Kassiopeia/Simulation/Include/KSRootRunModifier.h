#ifndef Kassiopeia_KSRootRunModifier_h_
#define Kassiopeia_KSRootRunModifier_h_

#include "KSEvent.h"
#include "KSList.h"
#include "KSParticle.h"
#include "KSRun.h"
#include "KSRunModifier.h"
#include "KSStep.h"
#include "KSTrack.h"

namespace Kassiopeia
{

class KSRootRunModifier : public KSComponentTemplate<KSRootRunModifier, KSRunModifier>
{
  public:
    KSRootRunModifier();
    KSRootRunModifier(const KSRootRunModifier& aCopy);
    KSRootRunModifier* Clone() const override;
    ~KSRootRunModifier() override;

    //**********
    // modifier
    //**********

  protected:
    bool ExecutePreRunModification(KSRun& aRun) override;
    bool ExecutePostRunModification(KSRun& aRun) override;

    //***********
    //composition
    //***********

  public:
    void AddModifier(KSRunModifier* aModifier);
    void RemoveModifier(KSRunModifier* aModifier);

  private:
    KSList<KSRunModifier> fModifiers;
    KSRunModifier* fModifier;

    //******
    //action
    //******

  public:
    void SetRun(KSRun* aRun);

    bool ExecutePreRunModification();
    bool ExecutePostRunModification();

    void PushUpdateComponent() override;
    void PushDeupdateComponent() override;

  private:
    KSRun* fRun;
};


}  // namespace Kassiopeia

#endif
