#ifndef Kassiopeia_KSTerminator_h_
#define Kassiopeia_KSTerminator_h_

#include "KSComponentTemplate.h"
#include "KSParticle.h"

namespace Kassiopeia
{

class KSTerminator : public KSComponentTemplate<KSTerminator>
{
  public:
    KSTerminator();
    ~KSTerminator() override;

  public:
    virtual void CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag) = 0;
    virtual void ExecuteTermination(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                                    KSParticleQueue& aQueue) const = 0;
};


}  // namespace Kassiopeia

#endif
