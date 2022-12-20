#ifndef Kassiopeia_KSTrajTrajectoryExact_h_
#define Kassiopeia_KSTrajTrajectoryExact_h_

#include "KGBall.hh"
#include "KGBallSupportSet.hh"
#include "KSList.h"
#include "KSTrajExactTypes.h"
#include "KSTrajectory.h"

namespace Kassiopeia
{

class KSTrajTrajectoryExact :
    public KSComponentTemplate<KSTrajTrajectoryExact, KSTrajectory>,
    public KSTrajExactDifferentiator
{
  public:
    KSTrajTrajectoryExact();
    KSTrajTrajectoryExact(const KSTrajTrajectoryExact& aCopy);
    KSTrajTrajectoryExact* Clone() const override;
    ~KSTrajTrajectoryExact() override;

  public:
    void SetIntegrator(KSTrajExactIntegrator* anIntegrator);
    void ClearIntegrator(KSTrajExactIntegrator* anIntegrator);

    void SetInterpolator(KSTrajExactInterpolator* anInterpolator);
    void ClearInterpolator(KSTrajExactInterpolator* anInterpolator);

    void AddTerm(KSTrajExactDifferentiator* aTerm);
    void RemoveTerm(KSTrajExactDifferentiator* aTerm);

    void AddControl(KSTrajExactControl* aControl);
    void RemoveControl(KSTrajExactControl* aControlSize);

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
    void Reset() override;
    void CalculateTrajectory(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                             katrin::KThreeVector& aCenter, double& aRadius, double& aTimeStep) override;
    void ExecuteTrajectory(const double& aTimeStep, KSParticle& anIntermediateParticle) const override;
    void GetPiecewiseLinearApproximation(const KSParticle& anInitialParticle, const KSParticle& /*aFinalParticle*/,
                                         std::vector<KSParticle>* intermediateParticleStates) const override;

    //********************
    //exact term interface
    //********************

  public:
    void Differentiate(double aTime, const KSTrajExactParticle& aValue,
                       KSTrajExactDerivative& aDerivative) const override;

  private:
    KSTrajExactParticle fInitialParticle;
    mutable KSTrajExactParticle fIntermediateParticle;
    mutable KSTrajExactParticle fFinalParticle;
    mutable KSTrajExactError fError;

    KSTrajExactIntegrator* fIntegrator;
    KSTrajExactInterpolator* fInterpolator;
    KSList<KSTrajExactDifferentiator> fTerms;
    KSList<KSTrajExactControl> fControls;

    //piecewise linear approximation
    double fPiecewiseTolerance;
    unsigned int fNMaxSegments;
    mutable KGeoBag::KGBallSupportSet<3> fBallSupport;
    mutable std::vector<KSTrajExactParticle> fIntermediateParticleStates;

    unsigned int fMaxAttempts;
};

}  // namespace Kassiopeia

#endif
