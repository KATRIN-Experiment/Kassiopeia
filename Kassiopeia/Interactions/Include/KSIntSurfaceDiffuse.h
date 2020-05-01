#ifndef Kassiopeia_KSIntSurfaceDiffuse_h_
#define Kassiopeia_KSIntSurfaceDiffuse_h_

#include "KField.h"
#include "KSSurfaceInteraction.h"

namespace Kassiopeia
{

class KSStep;

class KSIntSurfaceDiffuse : public KSComponentTemplate<KSIntSurfaceDiffuse, KSSurfaceInteraction>
{
  public:
    KSIntSurfaceDiffuse();
    KSIntSurfaceDiffuse(const KSIntSurfaceDiffuse& aCopy);
    KSIntSurfaceDiffuse* Clone() const override;
    ~KSIntSurfaceDiffuse() override;

  public:
    void ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aSecondaries) override;
    void ExecuteReflection(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                           KSParticleQueue& aSecondaries);
    void ExecuteTransmission(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                             KSParticleQueue& aSecondaries);

  public:
    void SetReflectionLossFraction(double frac)
    {
        fReflectionLossFraction = frac;
        fUseRelativeLoss = true;
    };
    double GetReflectionLossFraction() const
    {
        return fTransmissionLossFraction;
    };

    void SetTransmissionLossFraction(double frac)
    {
        fTransmissionLossFraction = frac;
        fUseRelativeLoss = true;
    };
    double GetTransmissionLossFraction() const
    {
        return fTransmissionLossFraction;
    };


    K_SET_GET(double, Probability)
    K_SET_GET(double, ReflectionLoss)
    K_SET_GET(double, TransmissionLoss)


  private:
    double fReflectionLossFraction;
    double fTransmissionLossFraction;
    bool fUseRelativeLoss;
};

}  // namespace Kassiopeia

#endif
