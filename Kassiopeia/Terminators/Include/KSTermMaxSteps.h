#ifndef Kassiopeia_KSTermMaxSteps_h_
#define Kassiopeia_KSTermMaxSteps_h_

#include "KSTerminator.h"

namespace Kassiopeia
{

class KSTermMaxSteps : public KSComponentTemplate<KSTermMaxSteps, KSTerminator>
{
  public:
    KSTermMaxSteps();
    KSTermMaxSteps(const KSTermMaxSteps& aCopy);
    KSTermMaxSteps* Clone() const override;
    ~KSTermMaxSteps() override;

  public:
    void CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag) override;
    void ExecuteTermination(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aParticleQueue) const override;

  public:
    void SetMaxSteps(const unsigned int& maxsteps);

  protected:
    void ActivateComponent() override;
    void DeactivateComponent() override;

  private:
    unsigned int fMaxSteps;
    unsigned int fSteps;
};

inline void KSTermMaxSteps::SetMaxSteps(const unsigned int& maxsteps)
{
    fMaxSteps = maxsteps;
    return;
}

}  // namespace Kassiopeia

#endif
