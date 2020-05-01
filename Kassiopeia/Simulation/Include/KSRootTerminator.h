#ifndef Kassiopeia_KSRootTerminator_h_
#define Kassiopeia_KSRootTerminator_h_

#include "KSList.h"
#include "KSStep.h"
#include "KSTerminator.h"

namespace Kassiopeia
{

class KSTrack;

class KSRootTerminator : public KSComponentTemplate<KSRootTerminator, KSTerminator>
{
  public:
    KSRootTerminator();
    KSRootTerminator(const KSRootTerminator& aCopy);
    KSRootTerminator* Clone() const override;
    ~KSRootTerminator() override;

    //**********
    //terminator
    //**********

  protected:
    void CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag) override;
    void ExecuteTermination(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aQueue) const override;

    //***********
    //composition
    //***********

  public:
    void AddTerminator(KSTerminator* aTerminator);
    void RemoveTerminator(KSTerminator* aTerminator);

  private:
    KSList<KSTerminator> fTerminators;
    KSTerminator* fTerminator;

    //******
    //action
    //******

  public:
    void SetStep(KSStep* aStep);

    void CalculateTermination();
    void ExecuteTermination();

    void PushUpdateComponent() override;
    void PushDeupdateComponent() override;

  private:
    KSStep* fStep;
    const KSParticle* fInitialParticle;
    KSParticle* fTerminatorParticle;
    KSParticle* fFinalParticle;
    KSParticleQueue* fParticleQueue;
};


}  // namespace Kassiopeia

#endif
