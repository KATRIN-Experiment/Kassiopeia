#include "KSNavSpace.h"

#include "KSNavigatorsMessage.h"

#include <limits>

using namespace std;

namespace Kassiopeia
{

KSNavSpace::KSNavSpace() :
    fEnterSplit(false),
    fExitSplit(false),
    fFailCheck(false),
    fCurrentTrajectory(nullptr),
    fCurrentSpace(nullptr),
    fParentSpace(nullptr),
    fParentSpaceAnchor(0., 0., 0.),
    fParentSpaceDistance(0.),
    fParentSpaceRecalculate(true),
    fChildSpace(nullptr),
    fChildSpaceAnchor(0., 0., 0.),
    fChildSpaceDistance(0.),
    fChildSpaceRecalculate(true),
    fParentSide(nullptr),
    fParentSideAnchor(0., 0., 0.),
    fParentSideDistance(0.),
    fParentSideRecalculate(true),
    fChildSide(nullptr),
    fChildSideAnchor(0., 0., 0.),
    fChildSideDistance(0.),
    fChildSideRecalculate(true),
    fChildSurface(nullptr),
    fLastStepSurface(nullptr),
    fChildSurfaceAnchor(0., 0., 0.),
    fChildSurfaceDistance(0.),
    fChildSurfaceRecalculate(true),
    fSpaceInsideCheck(true),
    fNavigationFail(false),
    fSolver(),
    fIntermediateParticle()
{}
KSNavSpace::KSNavSpace(const KSNavSpace& aCopy) :
    KSComponent(),
    fEnterSplit(aCopy.fEnterSplit),
    fExitSplit(aCopy.fExitSplit),
    fFailCheck(aCopy.fFailCheck),
    fCurrentTrajectory(aCopy.fCurrentTrajectory),
    fCurrentSpace(aCopy.fCurrentSpace),
    fParentSpace(aCopy.fParentSpace),
    fParentSpaceAnchor(aCopy.fParentSpaceAnchor),
    fParentSpaceDistance(aCopy.fParentSpaceDistance),
    fParentSpaceRecalculate(aCopy.fParentSpaceRecalculate),
    fChildSpace(aCopy.fChildSpace),
    fChildSpaceAnchor(aCopy.fChildSpaceAnchor),
    fChildSpaceDistance(aCopy.fChildSpaceDistance),
    fChildSpaceRecalculate(aCopy.fChildSpaceRecalculate),
    fParentSide(aCopy.fParentSide),
    fParentSideAnchor(aCopy.fParentSideAnchor),
    fParentSideDistance(aCopy.fParentSideDistance),
    fParentSideRecalculate(aCopy.fParentSideRecalculate),
    fChildSide(aCopy.fChildSide),
    fChildSideAnchor(aCopy.fChildSideAnchor),
    fChildSideDistance(aCopy.fChildSideDistance),
    fChildSideRecalculate(aCopy.fChildSideRecalculate),
    fChildSurface(aCopy.fChildSurface),
    fLastStepSurface(aCopy.fLastStepSurface),
    fChildSurfaceAnchor(aCopy.fChildSurfaceAnchor),
    fChildSurfaceDistance(aCopy.fChildSurfaceDistance),
    fChildSurfaceRecalculate(aCopy.fChildSurfaceRecalculate),
    fSpaceInsideCheck(aCopy.fSpaceInsideCheck),
    fNavigationFail(aCopy.fNavigationFail),
    fSolver(),
    fIntermediateParticle()
{}
KSNavSpace* KSNavSpace::Clone() const
{
    return new KSNavSpace(*this);
}
KSNavSpace::~KSNavSpace() {}

void KSNavSpace::SetEnterSplit(const bool& aEnterSplit)
{
    fEnterSplit = aEnterSplit;
    return;
}
const bool& KSNavSpace::GetEnterSplit() const
{
    return fEnterSplit;
}

void KSNavSpace::SetExitSplit(const bool& aExitSplit)
{
    fExitSplit = aExitSplit;
    return;
}
const bool& KSNavSpace::GetExitSplit() const
{
    return fExitSplit;
}

void KSNavSpace::SetFailCheck(const bool& aValue)
{
    fFailCheck = aValue;
    return;
}
const bool& KSNavSpace::GetFailCheck() const
{
    return fFailCheck;
}

void KSNavSpace::CalculateNavigation(const KSTrajectory& aTrajectory, const KSParticle& aTrajectoryInitialParticle,
                                     const KSParticle& aTrajectoryFinalParticle, const KThreeVector& aTrajectoryCenter,
                                     const double& aTrajectoryRadius, const double& aTrajectoryStep,
                                     KSParticle& aNavigationParticle, double& aNavigationStep, bool& aNavigationFlag)
{
    navmsg_debug("navigation space <" << this->GetName() << "> calculating navigation:" << eom);

    KSSpace* tCurrentSpace = aTrajectoryInitialParticle.GetCurrentSpace();
    KThreeVector tInitialPoint = aTrajectoryInitialParticle.GetPosition();
    KThreeVector tFinalPoint = aTrajectoryFinalParticle.GetPosition();

    bool tSpaceFlag = false;
    double tSpaceTime = numeric_limits<double>::max();
    fParentSpace = nullptr;
    fChildSpace = nullptr;

    bool tSideFlag = false;
    double tSideTime = numeric_limits<double>::max();
    fParentSide = nullptr;
    fChildSide = nullptr;

    bool tSurfaceFlag = false;
    double tSurfaceTime = numeric_limits<double>::max();
    fChildSurface = nullptr;

    double tTime = 0.0;
    double tDistance = 0.0;
    double tInitialIntersection = 0.0;
    double tFinalIntersection = 0.0;
    KSSpace* tSpace = nullptr;
    KSSurface* tSurface = nullptr;
    KSSide* tSide = nullptr;

    navmsg_debug("  in space <" << tCurrentSpace->GetName() << "> at <" << aTrajectoryCenter.X() << ", "
                                << aTrajectoryCenter.Y() << ", " << aTrajectoryCenter.Z() << "> with radius <"
                                << aTrajectoryRadius << ">" << eom);

    if (tCurrentSpace != fCurrentSpace) {
        fCurrentSpace = tCurrentSpace;
        fParentSideRecalculate = true;
        fChildSideRecalculate = true;
        fChildSurfaceRecalculate = true;
        fParentSpaceRecalculate = true;
        fChildSpaceRecalculate = true;
    }

    fCurrentTrajectory = &aTrajectory;

    //check if particle is inside the space it should be (only if fail check is activated)
    if (fFailCheck && fSpaceInsideCheck) {
        if (tCurrentSpace->Outside(tInitialPoint)) {
            navmsg(eWarning) << "initial point " << tInitialPoint << " of trajectory is not inside current space <"
                             << tCurrentSpace->GetName() << ">" << eom;
            fNavigationFail = true;
            aNavigationStep = 0.0;
            aNavigationFlag = true;
            return;
        }

        for (int tSpaceIndex = 0; tSpaceIndex < tCurrentSpace->GetSpaceCount(); tSpaceIndex++) {
            tSpace = tCurrentSpace->GetSpace(tSpaceIndex);
            if (tSpace->Outside(tInitialPoint) == false) {
                navmsg(eWarning) << "initial point " << tInitialPoint << " of trajectory should be in child space <"
                                 << tSpace->GetName() << ">, but is not" << eom;
                fNavigationFail = true;
                aNavigationStep = 0.0;
                aNavigationFlag = true;
                return;
            }
        }
    }

    //**********
    //space exit
    //**********

    if (fParentSpaceRecalculate == false) {
        double tExcursion = (aTrajectoryCenter - fParentSpaceAnchor).Magnitude() + aTrajectoryRadius;
        if (tExcursion >= fParentSpaceDistance) {
            navmsg_debug("  excursion from parent space anchor exceeds cached distance <" << fParentSpaceDistance << ">"
                                                                                          << eom);
            fParentSpaceRecalculate = true;
        }
    }
    if (fParentSpaceRecalculate == true) {
        fParentSpaceAnchor = aTrajectoryCenter;
        fParentSpaceDistance = numeric_limits<double>::max();
        navmsg_debug("  minimum distance to exit must be recalculated" << eom);

        do {
            tSpace = tCurrentSpace;

            // calculate the distance between the anchor and the exit space
            tDistance = (fParentSpaceAnchor - tSpace->Point(fParentSpaceAnchor)).Magnitude();
            navmsg_debug("    distance to parent space <" << tSpace->GetName() << "> is <" << tDistance << ">" << eom);

            // if the current distance is less than the current minimum distance replace the current minimum distance with the current distance
            if (tDistance < fParentSpaceDistance) {
                fParentSpaceDistance = tDistance;
            }

            // if this distance is greater than the trajectory radius, skip the parent space
            if (tDistance > aTrajectoryRadius) {
                navmsg_debug("    skipping parent space <"
                             << tSpace->GetName() << "> because distance is greater than trajectory radius" << eom);
                break;
            }

            // calculate initial intersection
            tInitialIntersection = (tInitialPoint - tSpace->Point(tInitialPoint)).Dot(tSpace->Normal(tInitialPoint));
            navmsg_debug("    initial intersection to parent space <" << tSpace->GetName() << "> is <"
                                                                      << tInitialIntersection << ">" << eom);

            // calculate final intersection
            tFinalIntersection = (tFinalPoint - tSpace->Point(tFinalPoint)).Dot(tSpace->Normal(tFinalPoint));
            navmsg_debug("    final intersection to parent space <" << tSpace->GetName() << "> is <"
                                                                    << tFinalIntersection << ">" << eom);

            if (tFinalIntersection < 0) {
                //final state is inside current space, no exit, skipping
                navmsg_debug("    skipping parent space <"
                             << tSpace->GetName() << "> because final intersection function sign is negative" << eom);
                break;
            }
            else if (tFinalIntersection > 0) {
                //initial state inside, final state outside, default case for exiting

                double tLowerBoundary = 0.0;
                if (tInitialIntersection > 0) {
                    // rare case that only happens if the space was just entered but the initial intersection is not zero but positive due to numerics
                    // and the space is smaller than the step size
                    // as two intersections exist for this case, the lower boundary is increased a little bit to get the second intersection,
                    // which is where the particle leaves the space
                    tLowerBoundary = aTrajectoryStep / 100.0;
                }
                // calculate intersection time
                fIntermediateParticle.SetCurrentSpace(tSpace);
                fSolver.Solve(KMathBracketingSolver::eBrent,
                              this,
                              &KSNavSpace::SpaceIntersectionFunction,
                              0.,
                              tLowerBoundary,
                              aTrajectoryStep,
                              tTime);
                navmsg_debug("    time to parent space <" << tSpace->GetName() << "> is <" << tTime << ">" << eom);
            }
            else if (tFinalIntersection == 0) {
                //final state exactly on border of parent space, very rare case for exiting
                tTime = aTrajectoryStep;
                navmsg_debug("    time to parent space <" << tSpace->GetName() << "> is <" << tTime << ">" << eom);
            }


#ifdef Kassiopeia_ENABLE_DEBUG
            // calculate intersection distance
            aTrajectory.ExecuteTrajectory(tTime, fIntermediateParticle);
            tDistance =
                (fIntermediateParticle.GetPosition() - tSpace->Point(fIntermediateParticle.GetPosition())).Magnitude();
            navmsg_debug("    distance of calculated crossing position to parent space <"
                         << tSpace->GetName() << "> is <" << tDistance << ">" << eom);
#endif

            tSpaceFlag = true;
            tSpaceTime = tTime;
            fParentSpace = tSpace;
            fChildSpace = nullptr;

        } while (false);

        fParentSpaceRecalculate = false;
        navmsg_debug("  minimum distance to exit is <" << fParentSpaceDistance << ">" << eom);
    }

    //***********
    //space enter
    //***********

    if (fChildSpaceRecalculate == false) {
        double tExcursion = (aTrajectoryCenter - fChildSpaceAnchor).Magnitude() + aTrajectoryRadius;
        if (tExcursion >= fChildSpaceDistance) {
            navmsg_debug("  excursion from enter anchor exceeds cached distance <" << fChildSpaceDistance << ">"
                                                                                   << eom);
            fChildSpaceRecalculate = true;
        }
    }
    if (fChildSpaceRecalculate == true) {
        fChildSpaceAnchor = aTrajectoryCenter;
        fChildSpaceDistance = numeric_limits<double>::max();
        navmsg_debug("  minimum distance to enter must be recalculated" << eom);

        for (int tSpaceIndex = 0; tSpaceIndex < tCurrentSpace->GetSpaceCount(); tSpaceIndex++) {
            tSpace = tCurrentSpace->GetSpace(tSpaceIndex);

            // calculate the distance between the anchor and the enter space
            tDistance = (fChildSpaceAnchor - tSpace->Point(fChildSpaceAnchor)).Magnitude();
            navmsg_debug("    distance to child space <" << tSpace->GetName() << "> is <" << tDistance << ">" << eom);

            // if the current distance is less than the current minimum distance replace the current minimum distance with the current distance
            if (tDistance < fChildSpaceDistance) {
                fChildSpaceDistance = tDistance;
            }

            // if this distance is greater than the trajectory radius, skip the child space
            if (tDistance > aTrajectoryRadius) {
                navmsg_debug("    skipping child space <"
                             << tSpace->GetName() << "> because distance is greater than trajectory radius" << eom);
                continue;
            }

            // calculate initial intersection
            tInitialIntersection = (tInitialPoint - tSpace->Point(tInitialPoint)).Dot(tSpace->Normal(tInitialPoint));
            navmsg_debug("    initial intersection to child space <" << tSpace->GetName() << "> is <"
                                                                     << tInitialIntersection << ">" << eom);

            // calculate final intersection
            tFinalIntersection = (tFinalPoint - tSpace->Point(tFinalPoint)).Dot(tSpace->Normal(tFinalPoint));
            navmsg_debug("    final intersection to child space <" << tSpace->GetName() << "> is <"
                                                                   << tFinalIntersection << ">" << eom);


            if (tFinalIntersection <= 0) {
                //final state inside, default case for enter
                //including zero intersection function, if final state hits exactly the border of the space to enter (very rare case)

                double tLowerBoundary = 0.0;
                if (tInitialIntersection <= 0) {
                    //very rare case where the space was just left and the initial state of the next step is still inside due to numerics
                    //increase the lower boundary to make sure not to get two intersections (which would results in gsl error for brent solver)
                    tLowerBoundary = aTrajectoryStep / 100.0;
                }

                // calculate intersection time
                fIntermediateParticle.SetCurrentSpace(tSpace);
                fSolver.Solve(KMathBracketingSolver::eBrent,
                              this,
                              &KSNavSpace::SpaceIntersectionFunction,
                              0.,
                              tLowerBoundary,
                              aTrajectoryStep,
                              tTime);
                navmsg_debug("    time to child space <" << tSpace->GetName() << "> is <" << tTime << ">" << eom);
            }
            else {
                //final state outside, particle did not enter space (or step size is to large!)
                navmsg_debug("    skipping child space <"
                             << tSpace->GetName() << "> because final intersection function is positive" << eom);
                continue;
            }

            // if the intersection time is not the smallest, skip the child space
            if (tTime > tSpaceTime) {
                navmsg_debug("    skipping child space <" << tSpace->GetName()
                                                          << "> because intersection time is not smallest" << eom);
                continue;
            }

#ifdef Kassiopeia_ENABLE_DEBUG
            // calculate intersection distance
            aTrajectory.ExecuteTrajectory(tTime, fIntermediateParticle);
            tDistance =
                (fIntermediateParticle.GetPosition() - tSpace->Point(fIntermediateParticle.GetPosition())).Magnitude();
            navmsg_debug("    distance of calculated crossing position to child space <"
                         << tSpace->GetName() << "> is <" << tDistance << ">" << eom);
#endif

            tSpaceFlag = true;
            tSpaceTime = tTime;
            fParentSpace = nullptr;
            fChildSpace = tSpace;
        }

        fChildSpaceRecalculate = false;
        navmsg_debug("  minimum distance to child spaces is <" << fChildSpaceDistance << ">" << eom);
    }

    //************
    //parent sides
    //************

    if (fParentSideRecalculate == false) {
        double tExcursion = (aTrajectoryCenter - fParentSideAnchor).Magnitude() + aTrajectoryRadius;
        if (tExcursion >= fParentSideDistance) {
            navmsg_debug("  excursion from parent side anchor exceeds cached distance <" << fParentSideDistance << ">"
                                                                                         << eom);

            fParentSideRecalculate = true;
        }
    }
    if (fParentSideRecalculate == true) {
        fParentSideAnchor = aTrajectoryCenter;
        fParentSideDistance = numeric_limits<double>::max();
        navmsg_debug("  minimum distance to parent sides must be recalculated" << eom);

        for (int tParentSideIndex = 0; tParentSideIndex < tCurrentSpace->GetSideCount(); tParentSideIndex++) {
            tSide = tCurrentSpace->GetSide(tParentSideIndex);

            // calculate the distance between the anchor and the parent side
            tDistance = (fParentSideAnchor - tSide->Point(fParentSideAnchor)).Magnitude();
            navmsg_debug("    distance to parent side <" << tSide->GetName() << "> is <" << tDistance << ">" << eom);

            // if the current distance is less than the current minimum distance replace the current minimum distance with the current distance
            if (tDistance < fParentSideDistance) {
                fParentSideDistance = tDistance;
            }

            // if this distance is greater than the trajectory radius, skip the parent side
            if (tDistance > aTrajectoryRadius) {
                navmsg_debug("    skipping parent side <"
                             << tSide->GetName() << "> because distance is greater than trajectory radius" << eom);
                continue;
            }

            // calculate initial intersection
            tInitialIntersection = (tInitialPoint - tSide->Point(tInitialPoint)).Dot(tSide->Normal(tInitialPoint));
            navmsg_debug("    initial intersection to parent side <" << tSide->GetName() << "> is <"
                                                                     << tInitialIntersection << ">" << eom);

            // calculate final intersection
            tFinalIntersection = (tFinalPoint - tSide->Point(tFinalPoint)).Dot(tSide->Normal(tFinalPoint));
            navmsg_debug("    final intersection to parent side <" << tSide->GetName() << "> is <" << tFinalIntersection
                                                                   << ">" << eom);

            if (tFinalIntersection < 0) {
                //final state is inside current space, no exit, skipping
                navmsg_debug("    skipping parent side <"
                             << tSide->GetName() << "> because final intersection function sign is negative" << eom);
                continue;
            }
            else if (tFinalIntersection > 0) {
                //initial state inside, final state outside, default case for exiting

                double tLowerBoundary = 0.0;
                if (tInitialIntersection > 0) {
                    // rare case that only happens if the space was just entered but the initial intersection is not zero but positive due to numerics
                    // and the space is smaller than the step size
                    // as two intersections exist for this case, the lower boundary is increased a little bit to get the second intersection,
                    // which is where the particle leaves the space
                    tLowerBoundary = aTrajectoryStep / 100.0;
                }
                // calculate intersection time
                fIntermediateParticle.SetCurrentSide(tSide);
                fSolver.Solve(KMathBracketingSolver::eBrent,
                              this,
                              &KSNavSpace::SideIntersectionFunction,
                              0.,
                              tLowerBoundary,
                              aTrajectoryStep,
                              tTime);
                navmsg_debug("    time to parent side <" << tSide->GetName() << "> is <" << tTime << ">" << eom);
            }
            else if (tFinalIntersection == 0) {
                //final state exactly on border of parent space, very rare case for exiting
                tTime = aTrajectoryStep;
                navmsg_debug("    time to parent space <" << tSpace->GetName() << "> is <" << tTime << ">" << eom);
            }


            // calculate intersection time
            fIntermediateParticle.SetCurrentSide(tSide);
            fSolver.Solve(KMathBracketingSolver::eBrent,
                          this,
                          &KSNavSpace::SideIntersectionFunction,
                          0.,
                          0.,
                          aTrajectoryStep,
                          tTime);
            navmsg_debug("    time to parent side <" << tSide->GetName() << "> is <" << tTime << ">" << eom);

            // if the intersection time is not the smallest, skip the parent side
            if (tTime > tSideTime) {
                navmsg_debug("    skipping parent side <" << tSide->GetName()
                                                          << "> because intersection time is not smallest" << eom);
                continue;
            }

#ifdef Kassiopeia_ENABLE_DEBUG
            // calculate intersection distance
            aTrajectory.ExecuteTrajectory(tTime, fIntermediateParticle);
            tDistance =
                (fIntermediateParticle.GetPosition() - tSide->Point(fIntermediateParticle.GetPosition())).Magnitude();
            navmsg_debug("    distance of calculated crossing position to parent side <" << tSide->GetName() << "> is <"
                                                                                         << tDistance << ">" << eom);
#endif

            tSideFlag = true;
            tSideTime = tTime;
            fParentSide = tSide;
            fChildSide = nullptr;
        }

        fParentSideRecalculate = false;
        navmsg_debug("  minimum distance to parent sides is <" << fParentSideDistance << ">" << eom);
    }

    //***********
    //child sides
    //***********

    if (fChildSideRecalculate == false) {
        double tExcursion = (aTrajectoryCenter - fChildSideAnchor).Magnitude() + aTrajectoryRadius;
        if (tExcursion >= fChildSideDistance) {
            navmsg_debug("  excursion from child side anchor exceeds cached distance <" << fChildSideDistance << ">"
                                                                                        << eom);

            fChildSideRecalculate = true;
        }
    }
    if (fChildSideRecalculate == true) {
        fChildSideAnchor = aTrajectoryCenter;
        fChildSideDistance = numeric_limits<double>::max();
        navmsg_debug("  minimum distance to child sides must be recalculated" << eom);

        for (int tSpaceIndex = 0; tSpaceIndex < tCurrentSpace->GetSpaceCount(); tSpaceIndex++) {
            tSpace = tCurrentSpace->GetSpace(tSpaceIndex);
            for (int tInternalSideIndex = 0; tInternalSideIndex < tSpace->GetSideCount(); tInternalSideIndex++) {
                tSide = tSpace->GetSide(tInternalSideIndex);

                // calculate the distance between the anchor and the child side
                tDistance = (fChildSideAnchor - tSide->Point(fChildSideAnchor)).Magnitude();
                navmsg_debug("    distance to child side <" << tSide->GetName() << "> is <" << tDistance << ">" << eom);

                // if the current distance is less than the current minimum distance replace the current minimum distance with the current distance
                if (tDistance < fChildSideDistance) {
                    fChildSideDistance = tDistance;
                }

                // if this distance is greater than the trajectory radius, skip the child side
                if (tDistance > aTrajectoryRadius) {
                    navmsg_debug("    skipping child side <"
                                 << tSide->GetName() << "> because distance is greater than trajectory radius" << eom);
                    continue;
                }

                // calculate initial intersection
                tInitialIntersection = (tInitialPoint - tSide->Point(tInitialPoint)).Dot(tSide->Normal(tInitialPoint));
                navmsg_debug("    initial intersection to child side <" << tSide->GetName() << "> is <"
                                                                        << tInitialIntersection << ">" << eom);

                // calculate final intersection
                tFinalIntersection = (tFinalPoint - tSide->Point(tFinalPoint)).Dot(tSide->Normal(tFinalPoint));
                navmsg_debug("    final intersection to child side <" << tSide->GetName() << "> is <"
                                                                      << tFinalIntersection << ">" << eom);


                if (tFinalIntersection <= 0) {
                    //final state inside, default case for enter
                    //including zero intersection function, if final state hits exactly the side of the space to enter (very rare case)

                    double tLowerBoundary = 0.0;
                    if (tInitialIntersection <= 0) {
                        //very rare case where the space was just left and the initial state of the next step is still inside due to numerics
                        //increase the lower boundary to make sure not to get two intersections (which would results in gsl error for brent solver)
                        tLowerBoundary = aTrajectoryStep / 100.0;
                    }

                    // calculate intersection time
                    fIntermediateParticle.SetCurrentSide(tSide);
                    fSolver.Solve(KMathBracketingSolver::eBrent,
                                  this,
                                  &KSNavSpace::SideIntersectionFunction,
                                  0.,
                                  tLowerBoundary,
                                  aTrajectoryStep,
                                  tTime);
                    navmsg_debug("    time to child side <" << tSide->GetName() << "> is <" << tTime << ">" << eom);
                }
                else {
                    //final state outside, particle did not enter space/side
                    navmsg_debug("    skipping child side <"
                                 << tSide->GetName() << "> because final intersection function is positive" << eom);
                    continue;
                }

                // if the intersection time is not the smallest, skip the child side
                if (tTime > tSideTime) {
                    navmsg_debug("    skipping child side <" << tSide->GetName()
                                                             << "> because intersection time is not smallest" << eom);
                    continue;
                }

#ifdef Kassiopeia_ENABLE_DEBUG
                // calculate intersection distance
                aTrajectory.ExecuteTrajectory(tTime, fIntermediateParticle);
                tDistance = (fIntermediateParticle.GetPosition() - tSide->Point(fIntermediateParticle.GetPosition()))
                                .Magnitude();
                navmsg_debug("    distance of calculated crossing position to child side <"
                             << tSide->GetName() << "> is <" << tDistance << ">" << eom);
#endif

                tSideFlag = true;
                tSideTime = tTime;
                fParentSide = nullptr;
                fChildSide = tSide;
            }
        }

        fChildSideRecalculate = false;
        navmsg_debug("  minimum distance to child sides is <" << fChildSideDistance << ">" << eom);
    }

    //**************
    //child surfaces
    //**************

    if (fChildSurfaceRecalculate == false) {
        double tExcursion = (aTrajectoryCenter - fChildSurfaceAnchor).Magnitude() + aTrajectoryRadius;
        if (tExcursion >= fChildSurfaceDistance) {
            fChildSurfaceRecalculate = true;

            navmsg_debug("  excursion from child surface anchor exceeds cached distance <" << fChildSurfaceDistance
                                                                                           << ">" << eom);
        }
    }
    if (fChildSurfaceRecalculate == true) {
        fChildSurfaceAnchor = aTrajectoryCenter;
        fChildSurfaceDistance = numeric_limits<double>::max();
        navmsg_debug("  minimum distance to child surfaces must be recalculated" << eom);

        for (int tSurfaceIndex = 0; tSurfaceIndex < tCurrentSpace->GetSurfaceCount(); tSurfaceIndex++) {
            tSurface = tCurrentSpace->GetSurface(tSurfaceIndex);

            // calculate the distance between the anchor and the child surface
            tDistance = (fChildSurfaceAnchor - tSurface->Point(fChildSurfaceAnchor)).Magnitude();
            navmsg_debug("    distance to child surface <" << tSurface->GetName() << "> is <" << tDistance << ">"
                                                           << eom);

            // if the current distance is less than the current minimum distance replace the current minimum distance with the current distance
            if (tDistance < fChildSurfaceDistance) {
                fChildSurfaceDistance = tDistance;
            }

            // if this distance is greater than the trajectory radius, skip the child surface
            if (tDistance > aTrajectoryRadius) {
                navmsg_debug("    skipping child surface <"
                             << tSurface->GetName() << "> because distance is greater than trajectory radius" << eom);
                continue;
            }

            // calculate initial intersection
            tInitialIntersection =
                (tInitialPoint - tSurface->Point(tInitialPoint)).Dot(tSurface->Normal(tInitialPoint));
            navmsg_debug("    initial intersection to child surface <" << tSurface->GetName() << "> is <"
                                                                       << tInitialIntersection << ">" << eom);

            // calculate final intersection
            tFinalIntersection = (tFinalPoint - tSurface->Point(tFinalPoint)).Dot(tSurface->Normal(tFinalPoint));
            navmsg_debug("    final intersection to child surface <" << tSurface->GetName() << "> is <"
                                                                     << tFinalIntersection << ">" << eom);


            if (tFinalIntersection == 0.) {
                //step hitting surface exactly, time is trajectory step
                tTime = aTrajectoryStep;
                navmsg_debug("    time to child surface <" << tSurface->GetName() << "> is <" << tTime << ">" << eom);
            }
            else if (tSurface == fLastStepSurface) {
                //we are starting on the surface, but if the trajectory is curved, we may cross the surface again with the same step
                KThreeVector tMomentum = aTrajectoryInitialParticle.GetMomentum();
                KThreeVector tNormal = tSurface->Normal(aTrajectoryInitialParticle.GetPosition());

                if (tMomentum.Dot(tNormal) > 0.) {
                    if (tFinalIntersection <= 0.) {
                        //particle turned around and crossed surface again
                        //find second crossing point (first one is at the start, increase lower boundary artificially
                        double tLowerBoundary = aTrajectoryStep / 100.0;
                        fIntermediateParticle.SetCurrentSurface(tSurface);
                        fSolver.Solve(KMathBracketingSolver::eBrent,
                                      this,
                                      &KSNavSpace::SurfaceIntersectionFunction,
                                      0.,
                                      tLowerBoundary,
                                      aTrajectoryStep,
                                      tTime);
                        navmsg_debug("    time to cross child surface <" << tSurface->GetName() << "> again is <"
                                                                         << tTime << ">" << eom);
                    }
                    else {
                        navmsg_debug("    skipping child surface <" << tSurface->GetName()
                                                                    << "> because it was crossed on last step" << eom);
                        continue;
                    }
                }
                else
                // tMommentum.Dot( tNormal ) < 0
                {
                    if (tFinalIntersection >= 0.) {
                        //particle turned around and crossed surface again
                        //find second crossing point (first one is at the start, increase lower boundary artificially
                        double tLowerBoundary = aTrajectoryStep / 100.0;
                        fIntermediateParticle.SetCurrentSurface(tSurface);
                        fSolver.Solve(KMathBracketingSolver::eBrent,
                                      this,
                                      &KSNavSpace::SurfaceIntersectionFunction,
                                      0.,
                                      tLowerBoundary,
                                      aTrajectoryStep,
                                      tTime);
                        navmsg_debug("    time to cross child surface <" << tSurface->GetName() << "> again is <"
                                                                         << tTime << ">" << eom);
                    }
                    else {
                        navmsg_debug("    skipping child surface <" << tSurface->GetName()
                                                                    << "> because it was crossed on last step" << eom);
                        continue;
                    }
                }
            }
            else if (tInitialIntersection == 0.0) {
                //starting exactly on surface, but it was not crossed before. The space was just entered or the simulation just started
                navmsg_debug("    skipping child surface <"
                             << tSurface->GetName()
                             << "> because initial particle state is exactly on surface (no crossing)" << eom);
                continue;
            }
            else if ((tInitialIntersection > 0. && tFinalIntersection < 0.) ||
                     (tInitialIntersection < 0. && tFinalIntersection > 0.)) {
                //sign change in intersection function indicates crossing of surface (default case)

                // calculate intersection time
                fIntermediateParticle.SetCurrentSurface(tSurface);
                fSolver.Solve(KMathBracketingSolver::eBrent,
                              this,
                              &KSNavSpace::SurfaceIntersectionFunction,
                              0.,
                              0.,
                              aTrajectoryStep,
                              tTime);
                navmsg_debug("    time to cross child surface <" << tSurface->GetName() << "> is <" << tTime << ">"
                                                                 << eom);
            }
            else {
                //signs are the same, no crossing
                navmsg_debug("    skipping child surface <"
                             << tSurface->GetName() << "> because intersection function signs are the same" << eom);
                continue;
            }

            // if the intersection time is not the smallest, skip the child surface
            if (tTime > tSurfaceTime) {
                navmsg_debug("    skipping child surface <" << tSurface->GetName()
                                                            << "> because intersection time is not smallest" << eom);
                continue;
            }

#ifdef Kassiopeia_ENABLE_DEBUG
            // calculate intersection distance
            aTrajectory.ExecuteTrajectory(tTime, fIntermediateParticle);
            tDistance = (fIntermediateParticle.GetPosition() - tSurface->Point(fIntermediateParticle.GetPosition()))
                            .Magnitude();
            navmsg_debug("    distance of calculated crossing position to child surface <"
                         << tSurface->GetName() << "> is <" << tDistance << ">" << eom);
#endif

            tSurfaceFlag = true;
            tSurfaceTime = tTime;
            fChildSurface = tSurface;
        }

        fChildSurfaceRecalculate = false;
        navmsg_debug("  minimum distance to child surfaces is <" << fChildSurfaceDistance << ">" << eom);
    }

    //reset last step surface
    fLastStepSurface = nullptr;

    //******************
    //case determination
    //******************

    if (tSideFlag == true) {
        if (tSideTime < tSurfaceTime) {
            fChildSurface = nullptr;
            fParentSpace = nullptr;
            fChildSpace = nullptr;

            aTrajectory.ExecuteTrajectory(tSideTime, aNavigationParticle);
            aNavigationStep = tSideTime;
            aNavigationFlag = true;

            navmsg_debug("  particle may cross side" << eom);

            return;
        }
    }

    if ((tSurfaceFlag == true) || (tSpaceFlag == true)) {
        if (tSurfaceTime < tSpaceTime) {
            fParentSpace = nullptr;
            fChildSpace = nullptr;

            aTrajectory.ExecuteTrajectory(tSurfaceTime, aNavigationParticle);
            aNavigationStep = tSurfaceTime;
            aNavigationFlag = true;

            navmsg_debug("  particle may cross surface" << eom);

            return;
        }
        else {
            fChildSurface = nullptr;

            aTrajectory.ExecuteTrajectory(tSpaceTime, aNavigationParticle);
            aNavigationStep = tSpaceTime;
            aNavigationFlag = true;

            navmsg_debug("  particle may change space" << eom);

            return;
        }
    }

    aNavigationParticle = aTrajectoryFinalParticle;
    aNavigationStep = aTrajectoryStep;
    aNavigationFlag = false;

    navmsg_debug("  no navigation occurred" << eom);

    //check if the navigator is in the correct state next step
    fSpaceInsideCheck = true;

    return;
}
void KSNavSpace::ExecuteNavigation(const KSParticle& aNavigationParticle, KSParticle& aFinalParticle,
                                   KSParticleQueue& aParticleQueue) const
{
    navmsg_debug("navigation space <" << this->GetName() << "> executing navigation:" << eom);

    //kill the particle if the navigation was wrong
    if (fNavigationFail) {
        fNavigationFail = false;
        aFinalParticle = aNavigationParticle;
        aFinalParticle.SetLabel(GetName());
        aFinalParticle.AddLabel("navigator_fail");
        aFinalParticle.SetActive(false);
        return;
    }
    //do not perform a space check next step, as it may lead to wrong results in the step directly after a space change, when the particle is on a boundary (due to numerical errors)
    fSpaceInsideCheck = false;

    if (fParentSpace != nullptr) {
        navmsg(eNormal) << "  parent space <" << fParentSpace->GetName() << "> was exited" << eom;

        aFinalParticle = aNavigationParticle;
        aFinalParticle.SetLabel(GetName());
        aFinalParticle.AddLabel(fParentSpace->GetName());
        aFinalParticle.AddLabel("exit");

        if (fExitSplit == true) {
            auto* tExitSplitParticle = new KSParticle(aFinalParticle);
            tExitSplitParticle->SetCurrentSpace(fParentSpace->GetParent());
            tExitSplitParticle->SetCurrentSurface(nullptr);
            tExitSplitParticle->SetCurrentSide(nullptr);
            tExitSplitParticle->ResetFieldCaching();
            aParticleQueue.push_back(tExitSplitParticle);
            aFinalParticle.SetActive(false);
        }

        return;
    }

    if (fChildSpace != nullptr) {
        navmsg(eNormal) << "  child space <" << fChildSpace->GetName() << "> was entered" << eom;

        aFinalParticle = aNavigationParticle;
        aFinalParticle.SetLabel(GetName());
        aFinalParticle.AddLabel(fChildSpace->GetName());
        aFinalParticle.AddLabel("enter");

        if (fEnterSplit == true) {
            auto* tEnterSplitParticle = new KSParticle(aFinalParticle);
            tEnterSplitParticle->SetCurrentSpace(fChildSpace);
            tEnterSplitParticle->SetCurrentSurface(nullptr);
            tEnterSplitParticle->SetCurrentSide(nullptr);
            tEnterSplitParticle->ResetFieldCaching();
            aParticleQueue.push_back(tEnterSplitParticle);
            aFinalParticle.SetActive(false);
        }

        return;
    }

    if (fParentSide != nullptr) {
        navmsg(eNormal) << "  parent side <" << fParentSide->GetName() << "> was crossed" << eom;

        aFinalParticle = aNavigationParticle;
        aFinalParticle.SetLabel(GetName());
        aFinalParticle.AddLabel(fParentSide->GetName());
        aFinalParticle.AddLabel("crossed");
        return;
    }

    if (fChildSide != nullptr) {
        navmsg(eNormal) << "  child side <" << fChildSide->GetName() << "> was crossed" << eom;

        aFinalParticle = aNavigationParticle;
        aFinalParticle.SetLabel(GetName());
        aFinalParticle.AddLabel(fChildSide->GetName());
        aFinalParticle.AddLabel("crossed");
        return;
    }

    if (fChildSurface != nullptr) {
        navmsg(eNormal) << "  child surface <" << fChildSurface->GetName() << "> was crossed" << eom;

        aFinalParticle = aNavigationParticle;
        aFinalParticle.SetLabel(GetName());
        aFinalParticle.AddLabel(fChildSurface->GetName());
        aFinalParticle.AddLabel("crossed");
        return;
    }

    navmsg(eError) << "could not determine space navigation" << eom;
    return;
}

void KSNavSpace::FinalizeNavigation(KSParticle& aFinalParticle) const
{
    navmsg_debug("navigation space <" << this->GetName() << "> finalizing navigation:" << eom);

    if (fParentSpace != nullptr) {
        navmsg_debug("  finalizing navigation for exiting of parent space <" << fParentSpace->GetName() << "> " << eom);

        aFinalParticle.SetCurrentSpace(fParentSpace->GetParent());
        aFinalParticle.ResetFieldCaching();
        fParentSpace->Exit();
        fParentSpace = nullptr;
        return;
    }

    if (fChildSpace != nullptr) {
        navmsg_debug("  finalizing navigation for entering of child space <" << fChildSpace->GetName() << "> " << eom);

        aFinalParticle.SetCurrentSpace(fChildSpace);
        aFinalParticle.ResetFieldCaching();
        fChildSpace->Enter();
        fChildSpace = nullptr;
        return;
    }

    if (fParentSide != nullptr) {
        navmsg_debug("  finalizing navigation for crossing of parent side <" << fParentSide->GetName() << "> " << eom);

        aFinalParticle.SetCurrentSide(fParentSide);
        aFinalParticle.ResetFieldCaching();
        fParentSide->On();
        fParentSide = nullptr;
        return;
    }

    if (fChildSide != nullptr) {
        navmsg_debug("  finalizing navigation for crossing of child side <" << fChildSide->GetName() << "> " << eom);

        aFinalParticle.SetCurrentSide(fChildSide);
        aFinalParticle.ResetFieldCaching();
        fChildSide->On();
        fChildSide = nullptr;
        return;
    }

    if (fChildSurface != nullptr) {
        navmsg_debug("  finalizing navigation for crossing of child surface <" << fChildSurface->GetName() << "> "
                                                                               << eom);

        fLastStepSurface = fChildSurface;
        aFinalParticle.SetCurrentSurface(fChildSurface);
        aFinalParticle.ResetFieldCaching();
        fChildSurface->On();
        fChildSurface = nullptr;
        return;
    }

    navmsg(eError) << "could not finalize space navigation" << eom;
    return;
}

void KSNavSpace::StartNavigation(KSParticle& aParticle, KSSpace* aRoot)
{
    navmsg_debug("navigation space <" << this->GetName() << "> starting navigation:" << eom);

    // reset navigation
    fCurrentSpace = nullptr;
    fParentSideRecalculate = true;
    fChildSideRecalculate = true;
    fChildSurfaceRecalculate = true;
    fParentSpaceRecalculate = true;
    fChildSpaceRecalculate = true;

    if (aParticle.GetCurrentSpace() == nullptr) {
        navmsg_debug("  computing fresh initial state" << eom);

        int tIndex = 0;
        KSSpace* tParentSpace = aRoot;
        KSSpace* tSpace = nullptr;
        while (tIndex < tParentSpace->GetSpaceCount()) {
            tSpace = tParentSpace->GetSpace(tIndex);
            if (tSpace->Outside(aParticle.GetPosition()) == false) {
                navmsg_debug("  activating space <" << tSpace->GetName() << ">" << eom);

                tSpace->Enter();
                tParentSpace = tSpace;
                tIndex = 0;
            }
            else {
                navmsg_debug("  skipping space <" << tSpace->GetName() << ">" << eom);

                tIndex++;
            }
        }

        aParticle.SetCurrentSpace(tParentSpace);
    }
    else {
        navmsg_debug("  entering given initial state" << eom);

        KSSpace* tSpace = aParticle.GetCurrentSpace();
        KSSurface* tSurface = aParticle.GetCurrentSurface();
        KSSide* tSide = aParticle.GetCurrentSide();
        deque<KSSpace*> tSequence;

        // get into the correct space state
        while (tSpace != aRoot) {
            tSequence.push_front(tSpace);
            tSpace = tSpace->GetParent();
        }
        for (auto tIt = tSequence.begin(); tIt != tSequence.end(); tIt++) {
            tSpace = *tIt;

            navmsg_debug("  entering space <" << tSpace->GetName() << ">" << eom);

            tSpace->Enter();
        }

        fLastStepSurface = aParticle.GetLastStepSurface();

        if (tSurface != nullptr) {
            navmsg_debug("  activating surface <" << tSurface->GetName() << ">" << eom);

            tSurface->On();
        }

        if (tSide != nullptr) {
            navmsg_debug("  activating side <" << tSide->GetName() << ">" << eom);

            tSide->On();
        }
    }

    return;
}
void KSNavSpace::StopNavigation(KSParticle& aParticle, KSSpace* aRoot)
{
    // reset navigation
    fCurrentSpace = nullptr;
    fParentSideRecalculate = true;
    fChildSideRecalculate = true;
    fChildSurfaceRecalculate = true;
    fParentSpaceRecalculate = true;
    fChildSpaceRecalculate = true;

    fParentSpace = nullptr;
    fChildSpace = nullptr;
    fParentSide = nullptr;
    fChildSide = nullptr;
    fChildSurface = nullptr;
    fLastStepSurface = nullptr;

    deque<KSSpace*> tSpaces;
    KSSpace* tSpace = aParticle.GetCurrentSpace();
    KSSurface* tSurface = aParticle.GetCurrentSurface();
    KSSide* tSide = aParticle.GetCurrentSide();

    // undo side state
    if (tSide != nullptr) {
        navmsg_debug("  deactivating side <" << tSide->GetName() << ">" << eom);

        tSide->Off();
    }

    // undo surface state
    if (tSurface != nullptr) {
        navmsg_debug("  deactivating surface <" << tSurface->GetName() << ">" << eom);

        tSurface->Off();
    }

    // undo space state
    while (tSpace != aRoot) {
        tSpaces.push_back(tSpace);
        tSpace = tSpace->GetParent();
    }
    for (auto tIt = tSpaces.begin(); tIt != tSpaces.end(); tIt++) {
        tSpace = *tIt;

        navmsg_debug("  deactivating space <" << tSpace->GetName() << ">" << eom);

        tSpace->Exit();
    }

    return;
}


double KSNavSpace::SpaceIntersectionFunction(const double& aTime)
{
    fCurrentTrajectory->ExecuteTrajectory(aTime, fIntermediateParticle);
    KThreeVector tParticlePoint = fIntermediateParticle.GetPosition();
    KThreeVector tSpacePoint = fIntermediateParticle.GetCurrentSpace()->Point(tParticlePoint);
    KThreeVector tSpaceNormal = fIntermediateParticle.GetCurrentSpace()->Normal(tParticlePoint);
    return (tParticlePoint - tSpacePoint).Dot(tSpaceNormal);
}
double KSNavSpace::SurfaceIntersectionFunction(const double& aTime)
{
    fCurrentTrajectory->ExecuteTrajectory(aTime, fIntermediateParticle);
    KThreeVector tParticlePoint = fIntermediateParticle.GetPosition();
    KThreeVector tSurfacePoint = fIntermediateParticle.GetCurrentSurface()->Point(tParticlePoint);
    KThreeVector tSurfaceNormal = fIntermediateParticle.GetCurrentSurface()->Normal(tParticlePoint);
    return (tParticlePoint - tSurfacePoint).Dot(tSurfaceNormal);
}
double KSNavSpace::SideIntersectionFunction(const double& aTime)
{
    fCurrentTrajectory->ExecuteTrajectory(aTime, fIntermediateParticle);
    KThreeVector tParticlePoint = fIntermediateParticle.GetPosition();
    KThreeVector tSidePoint = fIntermediateParticle.GetCurrentSide()->Point(tParticlePoint);
    KThreeVector tSideNormal = fIntermediateParticle.GetCurrentSide()->Normal(tParticlePoint);
    return (tParticlePoint - tSidePoint).Dot(tSideNormal);
}

}  // namespace Kassiopeia
