#ifndef Kassiopeia_KSRootTrackModifier_h_
#define Kassiopeia_KSRootTrackModifier_h_

#include "KSList.h"
#include "KSParticle.h"
#include "KSStep.h"
#include "KSTrack.h"
#include "KSTrackModifier.h"

namespace Kassiopeia
{

class KSRootTrackModifier : public KSComponentTemplate<KSRootTrackModifier, KSTrackModifier>
{
  public:
    KSRootTrackModifier();
    KSRootTrackModifier(const KSRootTrackModifier& aCopy);
    KSRootTrackModifier* Clone() const override;
    ~KSRootTrackModifier() override;

    //**********
    // modifier
    //**********

  protected:
    bool ExecutePreTrackModification(KSTrack& aTrack) override;
    bool ExecutePostTrackModification(KSTrack& aTrack) override;

    //***********
    //composition
    //***********

  public:
    void AddModifier(KSTrackModifier* aModifier);
    void RemoveModifier(KSTrackModifier* aModifier);

  private:
    KSList<KSTrackModifier> fModifiers;
    KSTrackModifier* fModifier;

    //******
    //action
    //******

  public:
    void SetTrack(KSTrack* aTrack);

    bool ExecutePreTrackModification();
    bool ExecutePostTrackModification();

    void PushUpdateComponent() override;
    void PushDeupdateComponent() override;

  private:
    KSTrack* fTrack;
};


}  // namespace Kassiopeia

#endif
