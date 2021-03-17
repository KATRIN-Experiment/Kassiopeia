#ifndef Kassiopeia_KSTrajTrajectoryMagnetic_h_
#define Kassiopeia_KSTrajTrajectoryMagnetic_h_

#include "KField.h"
#include "KGBall.hh"
#include "KGBallSupportSet.hh"
#include "KSList.h"
#include "KSTrajMagneticTypes.h"
#include "KSTrajectory.h"

namespace Kassiopeia
{

class KSTrajTrajectoryMagnetic :
    public KSComponentTemplate<KSTrajTrajectoryMagnetic, KSTrajectory>,
    public KSTrajMagneticDifferentiator
{
  public:
    KSTrajTrajectoryMagnetic();
    KSTrajTrajectoryMagnetic(const KSTrajTrajectoryMagnetic& aCopy);
    KSTrajTrajectoryMagnetic* Clone() const override;
    ~KSTrajTrajectoryMagnetic() override;

  public:
    void SetIntegrator(KSTrajMagneticIntegrator* anIntegrator);
    void ClearIntegrator(KSTrajMagneticIntegrator* anIntegrator);

    void SetInterpolator(KSTrajMagneticInterpolator* anInterpolator);
    void ClearInterpolator(KSTrajMagneticInterpolator* anInterpolator);

    void AddTerm(KSTrajMagneticDifferentiator* aTerm);
    void RemoveTerm(KSTrajMagneticDifferentiator* aTerm);

    void AddControl(KSTrajMagneticControl* aControl);
    void RemoveControl(KSTrajMagneticControl* aControl);

    void SetReverseDirection(const bool& aFlag);
    const bool& GetReverseDirection() const;

    void SetAttemptLimit(unsigned int n)
    {
        if (n > 1) {
            fMaxAttempts = n;
        }
        else {
            fMaxAttempts = 1;
        };
    }

    //**********
    //trajectory
    //**********

  public:
    void Reset() override;
    void CalculateTrajectory(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                             KGeoBag::KThreeVector& aCenter, double& aRadius, double& aTimeStep) override;
    void ExecuteTrajectory(const double& aTimeStep, KSParticle& anIntermediateParticle) const override;
    void GetPiecewiseLinearApproximation(const KSParticle& anInitialParticle, const KSParticle& /*aFinalParticle*/,
                                         std::vector<KSParticle>* intermediateParticleStates) const override;

    //********************
    //exact term interface
    //********************

    void SetPiecewiseTolerance(double ptol)
    {
        fPiecewiseTolerance = ptol;
    };
    double GetPiecewiseTolerance() const
    {
        return fPiecewiseTolerance;
    };

    void SetMaxNumberOfSegments(double n_max)
    {
        fNMaxSegments = n_max;
        if (fNMaxSegments < 1) {
            fNMaxSegments = 1;
        };
    };
    unsigned int GetMaxNumberOfSegments() const
    {
        return fNMaxSegments;
    };


  public:
    void Differentiate(double /*aTime*/, const KSTrajMagneticParticle& aValue,
                       KSTrajMagneticDerivative& aDerivative) const override;

  private:
    KSTrajMagneticParticle fInitialParticle;
    mutable KSTrajMagneticParticle fIntermediateParticle;
    mutable KSTrajMagneticParticle fFinalParticle;
    mutable KSTrajMagneticError fError;

    KSTrajMagneticIntegrator* fIntegrator;
    KSTrajMagneticInterpolator* fInterpolator;
    KSList<KSTrajMagneticDifferentiator> fTerms;
    KSList<KSTrajMagneticControl> fControls;

    //piecewise linear approximation
    double fPiecewiseTolerance;
    unsigned int fNMaxSegments;
    mutable KGeoBag::KGBallSupportSet<3> fBallSupport;
    mutable std::vector<KSTrajMagneticParticle> fIntermediateParticleStates;

    unsigned int fMaxAttempts;
};

}  // namespace Kassiopeia

#endif
