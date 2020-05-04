#ifndef Kassiopeia_KSNavSurface_h_
#define Kassiopeia_KSNavSurface_h_

#include "KSSurfaceNavigator.h"

namespace Kassiopeia
{

class KSNavSurface : public KSComponentTemplate<KSNavSurface, KSSurfaceNavigator>
{
  public:
    KSNavSurface();
    KSNavSurface(const KSNavSurface& aCopy);
    KSNavSurface* Clone() const override;
    ~KSNavSurface() override;

  public:
    void SetTransmissionSplit(const bool& aTransmissionSplit);
    const bool& GetTransmissionSplit() const;

    void SetReflectionSplit(const bool& aReflectionSplit);
    const bool& GetReflectionSplit() const;

  private:
    bool fTransmissionSplit;
    bool fReflectionSplit;

  public:
    void ExecuteNavigation(const KSParticle& anInitialParticle, const KSParticle& aNavigationParticle,
                           KSParticle& aFinalParticle, KSParticleQueue& aSecondaries) const override;
    void FinalizeNavigation(KSParticle& aFinalParticle) const override;
};

}  // namespace Kassiopeia

#endif
