#ifndef Kassiopeia_KSTrajTrajectoryLinear_h_
#define Kassiopeia_KSTrajTrajectoryLinear_h_

#include "KSTrajectory.h"

namespace Kassiopeia
{

class KSTrajTrajectoryLinear : public KSComponentTemplate<KSTrajTrajectoryLinear, KSTrajectory>
{
  public:
    KSTrajTrajectoryLinear();
    KSTrajTrajectoryLinear(const KSTrajTrajectoryLinear& aCopy);
    KSTrajTrajectoryLinear* Clone() const override;
    ~KSTrajTrajectoryLinear() override;

  public:
    void SetLength(const double& aLength);
    const double& GetLength() const;

  private:
    double fLength;

    //**********
    //trajectory
    //**********

  public:
    void Reset() override;
    void CalculateTrajectory(const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KThreeVector& aCenter,
                             double& aRadius, double& aTimeStep) override;
    void ExecuteTrajectory(const double& aTimeStep, KSParticle& anIntermediateParticle) const override;
    void GetPiecewiseLinearApproximation(const KSParticle& anInitialParticle, const KSParticle& aFinalParticle,
                                         std::vector<KSParticle>* intermediateParticleStates) const override;


  private:
    double fTime;
    KThreeVector fPosition;
    KThreeVector fVelocity;

    //internal state for piecewise approximation
    KSParticle fFirstParticle;
    KSParticle fLastParticle;
};

}  // namespace Kassiopeia

#endif
