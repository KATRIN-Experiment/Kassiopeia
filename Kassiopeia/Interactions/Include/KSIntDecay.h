#ifndef Kassiopeia_KSIntDecay_h_
#define Kassiopeia_KSIntDecay_h_

#include "KSIntDecayCalculator.h"
#include "KSSpaceInteraction.h"

#include <vector>
using std::vector;

namespace Kassiopeia
{

class KSIntDecayCalculator;

class KSIntDecay : public KSComponentTemplate<KSIntDecay, KSSpaceInteraction>
{
  public:
    KSIntDecay();
    KSIntDecay(const KSIntDecay& aCopy);
    KSIntDecay* Clone() const override;
    ~KSIntDecay() override;

  public:
    std::vector<double> CalculateLifetimes(const KSParticle& aTrajectoryInitialParticle);

    void CalculateInteraction(const KSTrajectory& aTrajectory, const KSParticle& aTrajectoryInitialParticle,
                              const KSParticle& aTrajectoryFinalParticle, const KThreeVector& aTrajectoryCenter,
                              const double& aTrajectoryRadius, const double& aTrajectoryTimeStep,
                              KSParticle& anInteractionParticle, double& aTimeStep, bool& aFlag) override;

    void ExecuteInteraction(const KSParticle& anInteractionParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aSecondaries) const override;

    //***********
    //composition
    //***********

  public:
    void SetSplit(const bool& aSplit);
    const bool& GetSplit() const;

    void AddCalculator(KSIntDecayCalculator* const aScatteringCalculator);
    void RemoveCalculator(KSIntDecayCalculator* const aScatteringCalculator);

    void SetEnhancement(double anEnhancement);

  private:
    bool fSplit;
    KSIntDecayCalculator* fCalculator;
    std::vector<KSIntDecayCalculator*> fCalculators;
    std::vector<double> fLifeTimes;

    double fEnhancement;

    //**************
    //initialization
    //**************

  protected:
    void InitializeComponent() override;
    void ActivateComponent() override;
    void DeinitializeComponent() override;
    void DeactivateComponent() override;
    void PushUpdateComponent() override;
    void PushDeupdateComponent() override;
};

}  // namespace Kassiopeia

#endif
