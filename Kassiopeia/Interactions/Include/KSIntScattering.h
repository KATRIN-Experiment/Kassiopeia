#ifndef Kassiopeia_KSIntScattering_h_
#define Kassiopeia_KSIntScattering_h_

#include "KSIntCalculator.h"
#include "KSIntDensity.h"
#include "KSSpaceInteraction.h"

#include <vector>
using std::vector;

namespace Kassiopeia
{

class KSIntCalculator;

class KSIntScattering : public KSComponentTemplate<KSIntScattering, KSSpaceInteraction>
{
  public:
    KSIntScattering();
    KSIntScattering(const KSIntScattering& aCopy);
    KSIntScattering* Clone() const override;
    ~KSIntScattering() override;

  public:
    void CalculateAverageCrossSection(const KSParticle& aTrajectoryInitialParticle,
                                      const KSParticle& aTrajectoryFinalParticle, double& anAverageCrossSection);

    void DiceCalculator(const double& anAverageCrossSection);

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

    void SetDensity(KSIntDensity* const aDensityCalculator);
    void ClearDensity(KSIntDensity* const aDensityCalculator);

    void AddCalculator(KSIntCalculator* const aScatteringCalculator);
    void RemoveCalculator(KSIntCalculator* const aScatteringCalculator);

    void SetEnhancement(double anEnhancement);

  private:
    bool fSplit;
    KSIntDensity* fDensity;
    KSIntCalculator* fCalculator;
    std::vector<KSIntCalculator*> fCalculators;
    std::vector<double> fCrossSections;

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
