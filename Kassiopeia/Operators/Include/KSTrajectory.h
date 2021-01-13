#ifndef Kassiopeia_KSTrajectory_h_
#define Kassiopeia_KSTrajectory_h_

#include "KSComponentTemplate.h"
#include "KSParticle.h"

#include <vector>

namespace Kassiopeia
{

class KSTrajectory : public KSComponentTemplate<KSTrajectory>
{
  public:
    KSTrajectory();
    ~KSTrajectory() override;

  public:
    //clear any internal state data (caching)
    virtual void Reset()
    {
        ;
    };

    virtual void CalculateTrajectory(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                                     KGeoBag::KThreeVector& aCenter, double& aRadius, double& aTimeStep) = 0;

    virtual void ExecuteTrajectory(const double& aTimeStep, KSParticle& anIntermediateParticle) const = 0;

    virtual void GetPiecewiseLinearApproximation(const KSParticle& anInitialParticle, const KSParticle& aFinalParticle,
                                                 std::vector<KSParticle>* intermediateParticleStates) const = 0;

    //we are forced to use a static function because this is accessed
    //as a callback from a c-function  (gsl error handler)
    //this callback is not thread safe. However, that being said,
    //what situation would require running more than one instance of kassiopeia/trajectory??
    //we could get around this ugly hack if we could use C++11 std::bind
    static void ClearAbort()
    {
        fAbortSignal = false;
    };
    static void SetAbort()
    {
        fAbortSignal = true;
    };

  protected:
    static bool fAbortSignal;
};

}  // namespace Kassiopeia

#endif
