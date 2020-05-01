#ifndef Kassiopeia_KSTermSecondaries_h_
#define Kassiopeia_KSTermSecondaries_h_

#include "KSTerminator.h"

namespace Kassiopeia
{

class KSTermSecondaries : public KSComponentTemplate<KSTermSecondaries, KSTerminator>
{
  public:
    KSTermSecondaries();
    KSTermSecondaries(const KSTermSecondaries& aCopy);
    KSTermSecondaries* Clone() const override;
    ~KSTermSecondaries() override;

  public:
    void CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag) override;
    void ExecuteTermination(const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue&) const override;
};

}  // namespace Kassiopeia

#endif
