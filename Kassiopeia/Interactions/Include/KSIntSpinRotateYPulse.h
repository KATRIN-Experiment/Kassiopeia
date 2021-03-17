#ifndef Kassiopeia_KSIntSpinRotateYPulse_h_
#define Kassiopeia_KSIntSpinRotateYPulse_h_

#include "KSSpaceInteraction.h"

#include <vector>

namespace Kassiopeia
{

class KSIntSpinRotateYPulse : public KSComponentTemplate<KSIntSpinRotateYPulse, KSSpaceInteraction>
{
  public:
    KSIntSpinRotateYPulse();
    KSIntSpinRotateYPulse(const KSIntSpinRotateYPulse& aCopy);
    KSIntSpinRotateYPulse* Clone() const override;
    ~KSIntSpinRotateYPulse() override;

  public:
    void CalculateInteraction(const KSTrajectory& aTrajectory, const KSParticle& aTrajectoryInitialParticle,
                              const KSParticle& aTrajectoryFinalParticle,
                              const KGeoBag::KThreeVector& aTrajectoryCenter, const double& aTrajectoryRadius,
                              const double& aTrajectoryTimeStep, KSParticle& anInteractionParticle, double& aTimeStep,
                              bool& aFlag) override;

    void ExecuteInteraction(const KSParticle& anInteractionParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aSecondaries) const override;

    //***********
    //composition
    //***********

  public:
    void SetTime(const double& aTime);
    void SetAngle(const double& anAngle);
    void SetIsAdiabatic(const bool& anIsAdiabatic);

  private:
    mutable bool fDone;
    double fTime;
    double fAngle;
    bool fIsAdiabatic;

    //**************
    //initialization
    //**************

    // protected:
    //     virtual void PushUpdateComponent();
    //     virtual void PushDeupdateComponent();
};

}  // namespace Kassiopeia

#endif
