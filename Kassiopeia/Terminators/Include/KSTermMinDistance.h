#ifndef Kassiopeia_KSTermMinDistance_h_
#define Kassiopeia_KSTermMinDistance_h_

#include "KField.h"
#include "KGCore.hh"
#include "KSTerminator.h"

namespace Kassiopeia
{

class KSParticle;

class KSTermMinDistance : public KSComponentTemplate<KSTermMinDistance, KSTerminator>
{
  public:
    KSTermMinDistance();
    KSTermMinDistance(const KSTermMinDistance& aCopy);
    KSTermMinDistance* Clone() const override;
    ~KSTermMinDistance() override;

  public:
    void CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag) override;
    void ExecuteTermination(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aParticleQueue) const override;

    void AddSurface(KGeoBag::KGSurface* aSurface);
    void AddSpace(KGeoBag::KGSpace* aSpace);

  protected:
    void InitializeComponent() override;


  private:
    ;
    K_GET(double, MinDistancePerStep);
    ;
    K_SET(double, MinDistance);
    std::vector<KGeoBag::KGSurface*> fSurfaces;
    std::vector<KGeoBag::KGSpace*> fSpaces;
};
}  // namespace Kassiopeia

#endif
