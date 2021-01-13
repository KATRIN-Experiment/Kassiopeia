#ifndef Kassiopeia_KSNavSpace_h_
#define Kassiopeia_KSNavSpace_h_

#include "KMathBracketingSolver.h"
#include "KSSpaceNavigator.h"

namespace Kassiopeia
{

class KSNavSpace : public KSComponentTemplate<KSNavSpace, KSSpaceNavigator>
{
  public:
    KSNavSpace();
    KSNavSpace(const KSNavSpace& aCopy);
    KSNavSpace* Clone() const override;
    ~KSNavSpace() override;

  public:
    void SetEnterSplit(const bool& aEnterSplit);
    const bool& GetEnterSplit() const;

    void SetExitSplit(const bool& aExitSplit);
    const bool& GetExitSplit() const;

    void SetFailCheck(const bool& aValue);
    const bool& GetFailCheck() const;

  private:
    bool fEnterSplit;
    bool fExitSplit;
    bool fFailCheck;

  public:
    void CalculateNavigation(const KSTrajectory& aTrajectory, const KSParticle& aTrajectoryInitialParticle,
                             const KSParticle& aTrajectoryFinalParticle, const KGeoBag::KThreeVector& aTrajectoryCenter,
                             const double& aTrajectoryRadius, const double& aTrajectoryStep,
                             KSParticle& aNavigationParticle, double& aNavigationStep, bool& aNavigationFlag) override;
    void ExecuteNavigation(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                           KSParticleQueue& aSecondaries) const override;
    void FinalizeNavigation(KSParticle& aFinalParticle) const override;
    void StartNavigation(KSParticle& aParticle, KSSpace* aRoot) override;
    void StopNavigation(KSParticle& aParticle, KSSpace* aRoot) override;

  private:
    const KSTrajectory* fCurrentTrajectory;
    const KSSpace* fCurrentSpace;

    mutable KSSpace* fParentSpace;
    KGeoBag::KThreeVector fParentSpaceAnchor;
    double fParentSpaceDistance;
    bool fParentSpaceRecalculate;

    mutable KSSpace* fChildSpace;
    KGeoBag::KThreeVector fChildSpaceAnchor;
    double fChildSpaceDistance;
    bool fChildSpaceRecalculate;

    mutable KSSide* fParentSide;
    KGeoBag::KThreeVector fParentSideAnchor;
    double fParentSideDistance;
    bool fParentSideRecalculate;

    mutable KSSide* fChildSide;
    KGeoBag::KThreeVector fChildSideAnchor;
    double fChildSideDistance;
    bool fChildSideRecalculate;

    mutable KSSurface* fChildSurface;
    mutable KSSurface* fLastStepSurface;
    KGeoBag::KThreeVector fChildSurfaceAnchor;
    double fChildSurfaceDistance;
    bool fChildSurfaceRecalculate;

    mutable bool fSpaceInsideCheck;
    mutable bool fNavigationFail;

    double SpaceIntersectionFunction(const double& anIntersection);
    double SurfaceIntersectionFunction(const double& anIntersection);
    double SideIntersectionFunction(const double& anIntersection);

    katrin::KMathBracketingSolver fSolver;
    KSParticle fIntermediateParticle;
};

}  // namespace Kassiopeia

#endif
