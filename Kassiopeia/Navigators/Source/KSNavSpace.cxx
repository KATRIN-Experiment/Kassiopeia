#include "KSNavSpace.h"
#include "KSNavigatorsMessage.h"

#include <limits>
using std::numeric_limits;

namespace Kassiopeia
{

    KSNavSpace::KSNavSpace() :
            fEnterSplit( false ),
            fExitSplit( false ),
            fTolerance( 1.e-10 ),
            fCurrentTrajectory( NULL ),
            fCurrentSpace( NULL ),
            fParentSpace( NULL ),
            fParentSpaceAnchor( 0., 0., 0. ),
            fParentSpaceDistance( 0. ),
            fParentSpaceRecalculate( true ),
            fChildSpace( NULL ),
            fChildSpaceAnchor( 0., 0., 0. ),
            fChildSpaceDistance( 0. ),
            fChildSpaceRecalculate( true ),
            fParentSide( NULL ),
            fParentSideAnchor( 0., 0., 0. ),
            fParentSideDistance( 0. ),
            fParentSideRecalculate( true ),
            fChildSide( NULL ),
            fChildSideAnchor( 0., 0., 0. ),
            fChildSideDistance( 0. ),
            fChildSideRecalculate( true ),
            fChildSurface( NULL ),
            fChildSurfaceAnchor( 0., 0., 0. ),
            fChildSurfaceDistance( 0. ),
            fChildSurfaceRecalculate( true ),
            fSolver(),
            fIntermediateParticle()
    {
    }
    KSNavSpace::KSNavSpace( const KSNavSpace& aCopy ) :
            fEnterSplit( aCopy.fEnterSplit ),
            fExitSplit( aCopy.fExitSplit ),
            fTolerance( aCopy.fTolerance ),
            fCurrentTrajectory( aCopy.fCurrentTrajectory ),
            fCurrentSpace( aCopy.fCurrentSpace ),
            fParentSpace( aCopy.fParentSpace ),
            fParentSpaceAnchor( aCopy.fParentSpaceAnchor ),
            fParentSpaceDistance( aCopy.fParentSpaceDistance ),
            fParentSpaceRecalculate( aCopy.fParentSpaceRecalculate ),
            fChildSpace( aCopy.fChildSpace ),
            fChildSpaceAnchor( aCopy.fChildSpaceAnchor ),
            fChildSpaceDistance( aCopy.fChildSpaceDistance ),
            fChildSpaceRecalculate( aCopy.fChildSpaceRecalculate ),
            fParentSide( aCopy.fParentSide ),
            fParentSideAnchor( aCopy.fParentSideAnchor ),
            fParentSideDistance( aCopy.fParentSideDistance ),
            fParentSideRecalculate( aCopy.fParentSideRecalculate ),
            fChildSide( aCopy.fChildSide ),
            fChildSideAnchor( aCopy.fChildSideAnchor ),
            fChildSideDistance( aCopy.fChildSideDistance ),
            fChildSideRecalculate( aCopy.fChildSideRecalculate ),
            fChildSurface( aCopy.fChildSurface ),
            fChildSurfaceAnchor( aCopy.fChildSurfaceAnchor ),
            fChildSurfaceDistance( aCopy.fChildSurfaceDistance ),
            fChildSurfaceRecalculate( aCopy.fChildSurfaceRecalculate ),
            fSolver(),
            fIntermediateParticle()
    {
    }
    KSNavSpace* KSNavSpace::Clone() const
    {
        return new KSNavSpace( *this );
    }
    KSNavSpace::~KSNavSpace()
    {
    }

    void KSNavSpace::SetEnterSplit( const bool& aEnterSplit )
    {
        fEnterSplit = aEnterSplit;
        return;
    }
    const bool& KSNavSpace::GetEnterSplit() const
    {
        return fEnterSplit;
    }

    void KSNavSpace::SetExitSplit( const bool& aExitSplit )
    {
        fExitSplit = aExitSplit;
        return;
    }
    const bool& KSNavSpace::GetExitSplit() const
    {
        return fExitSplit;
    }

    void KSNavSpace::SetTolerance( const double& aTolerance )
    {
        fTolerance = aTolerance;
        return;
    }
    const double& KSNavSpace::GetTolerance() const
    {
        return fTolerance;
    }

    void KSNavSpace::CalculateNavigation( const KSTrajectory& aTrajectory, const KSParticle& aTrajectoryInitialParticle, const KSParticle& aTrajectoryFinalParticle, const KThreeVector& aTrajectoryCenter, const double& aTrajectoryRadius, const double& aTrajectoryStep, KSParticle& aNavigationParticle, double& aNavigationStep, bool& aNavigationFlag )
    {
        navmsg_debug( "navigation space <" << this->GetName() << "> calculating navigation:" << eom );

        KSSpace* tCurrentSpace = aTrajectoryInitialParticle.GetCurrentSpace();
        KThreeVector tInitialPoint = aTrajectoryInitialParticle.GetPosition();
        KThreeVector tFinalPoint = aTrajectoryFinalParticle.GetPosition();

        bool tSpaceFlag = false;
        double tSpaceTime = aTrajectoryStep;
        fParentSpace = NULL;
        fChildSpace = NULL;

        bool tSideFlag = false;
        double tSideTime = aTrajectoryStep;
        fParentSide = NULL;
        fChildSide = NULL;

        bool tSurfaceFlag = false;
        double tSurfaceTime = aTrajectoryStep;
        fChildSurface = NULL;

        double tTime;
        double tDistance;
        double tInitialIntersection;
        double tFinalIntersection;
        KSSpace* tSpace;
        KSSurface* tSurface;
        KSSide* tSide;

        navmsg_debug( "  in space <" << tCurrentSpace->GetName() << "> at <" << aTrajectoryCenter.X() << ", " << aTrajectoryCenter.Y() << ", " << aTrajectoryCenter.Z() << "> with radius <" << aTrajectoryRadius << ">" << eom );

        if( tCurrentSpace != fCurrentSpace )
        {
            fCurrentSpace = tCurrentSpace;
            fParentSideRecalculate = true;
            fChildSideRecalculate = true;
            fChildSurfaceRecalculate = true;
            fParentSpaceRecalculate = true;
            fChildSpaceRecalculate = true;
        }

        fCurrentTrajectory = &aTrajectory;

        //**********
        //space exit
        //**********

        if( fParentSpaceRecalculate == false )
        {
            double tExcursion = (aTrajectoryCenter - fParentSpaceAnchor).Magnitude() + aTrajectoryRadius;
            if( tExcursion > fParentSpaceDistance )
            {
                navmsg_debug( "  excursion from parent space anchor exceeds cached distance <" << fParentSpaceDistance << ">" << eom );
                fParentSpaceRecalculate = true;
            }
        }
        if( fParentSpaceRecalculate == true )
        {
            fParentSpaceAnchor = aTrajectoryCenter;
            fParentSpaceDistance = numeric_limits< double >::max();
            navmsg_debug( "  minimum distance to exit must be recalculated" << eom );

            do
            {
                tSpace = tCurrentSpace;

                // calculate the distance between the anchor and the enter space
                tDistance = (fParentSpaceAnchor - tSpace->Point( fParentSpaceAnchor )).Magnitude();
                navmsg_debug( "    distance to parent space <" << tSpace->GetName() << "> is <" << tDistance << ">" << eom );

                // if the current distance is less than the current minimum distance replace the current minimum distance with the current distance
                if( tDistance < fParentSpaceDistance )
                {
                    fParentSpaceDistance = tDistance;
                }

                // if this distance is greater than the trajectory radius, skip the parent space
                if( tDistance > aTrajectoryRadius )
                {
                    navmsg_debug( "    skipping parent space <" << tSpace->GetName() << "> because distance is greater than trajectory radius" << eom );
                    break;
                }

                // examine intersection function
                fIntermediateParticle.SetCurrentSpace( tSpace );

                // calculate initial intersection
                tInitialIntersection = ( tInitialPoint - tSpace->Point( tInitialPoint )).Dot( tSpace->Normal( tInitialPoint ) );
                navmsg_debug( "    initial intersection to parent space <" << tSpace->GetName() << "> is <" << tInitialIntersection << ">" << eom );

                // calculate final intersection
                tFinalIntersection = ( tFinalPoint - tSpace->Point( tFinalPoint )).Dot( tSpace->Normal( tFinalPoint ) );
                navmsg_debug( "    final intersection to parent space <" << tSpace->GetName() << "> is <" << tFinalIntersection << ">" << eom );

                // if the initial intersection function is within tolerance, skip the parent space
                if( fabs( tInitialIntersection / aTrajectoryRadius ) < fTolerance )
                {
                    navmsg_debug( "    skipping parent space <" << tSpace->GetName() << "> because intersection function is within tolerance" << eom );
                    break;
                }

                // if the intersection function signs are the same, skip the parent space
                if( (tInitialIntersection < 0.) == (tFinalIntersection < 0.) )
                {
                    navmsg_debug( "    skipping parent space <" << tSpace->GetName() << "> because intersection function signs are the same" << eom );
                    break;
                }

                // calculate intersection time
                fIntermediateParticle.SetCurrentSpace( tSpace );
                fSolver.Solve( KMathBracketingSolver::eBrent, this, &KSNavSpace::SpaceIntersectionFunction, 0., 0., aTrajectoryStep, tTime );
                navmsg_debug( "    time to parent space <" << tSpace->GetName() << "> is <" << tTime << ">" << eom );

                // if the intersection time is not the smallest, skip the parent space
                if( tTime > tSpaceTime )
                {
                    navmsg_debug( "    skipping parent space <" << tSpace->GetName() << "> because intersection time is not smallest" << eom );
                    break;
                }

                // calculate intersection distance
                aTrajectory.ExecuteTrajectory( tTime, fIntermediateParticle );
                tDistance = (fIntermediateParticle.GetPosition() - tSpace->Point( fIntermediateParticle.GetPosition() )).Magnitude();
                navmsg_debug( "    distance to parent space <" << tSpace->GetName() << "> is <" << tDistance << ">" << eom );

                // if the intersection distance is outside of tolerance, skip the parent space
                if( (tDistance / aTrajectoryRadius) > fTolerance )
                {
                    navmsg_debug( "    skipping parent space <" << tSpace->GetName() << "> because intersection distance is outside of tolerance" << eom );
                    break;
                }

                tSpaceFlag = true;
                tSpaceTime = tTime;
                fParentSpace = tSpace;
                fChildSpace = NULL;

            }
            while( false );

            fParentSpaceRecalculate = false;
            navmsg_debug( "  minimum distance to exit is <" << fParentSpaceDistance << ">" << eom );
        }

        //***********
        //space enter
        //***********

        if( fChildSpaceRecalculate == false )
        {
            double tExcursion = (aTrajectoryCenter - fChildSpaceAnchor).Magnitude() + aTrajectoryRadius;
            if( tExcursion > fChildSpaceDistance )
            {
                navmsg_debug( "  excursion from enter anchor exceeds cached distance <" << fChildSpaceDistance << ">" << eom );
                fChildSpaceRecalculate = true;
            }
        }
        if( fChildSpaceRecalculate == true )
        {
            fChildSpaceAnchor = aTrajectoryCenter;
            fChildSpaceDistance = numeric_limits< double >::max();
            navmsg_debug( "  minimum distance to enter must be recalculated" << eom );

            for( int tSpaceIndex = 0; tSpaceIndex < tCurrentSpace->GetSpaceCount(); tSpaceIndex++ )
            {
                tSpace = tCurrentSpace->GetSpace( tSpaceIndex );

                // calculate the distance between the anchor and the enter space
                tDistance = (fChildSpaceAnchor - tSpace->Point( fChildSpaceAnchor )).Magnitude();
                navmsg_debug( "    distance to child space <" << tSpace->GetName() << "> is <" << tDistance << ">" << eom );

                // if the current distance is less than the current minimum distance replace the current minimum distance with the current distance
                if( tDistance < fChildSpaceDistance )
                {
                    fChildSpaceDistance = tDistance;
                }

                // if this distance is greater than the trajectory radius, skip the child space
                if( tDistance > aTrajectoryRadius )
                {
                    navmsg_debug( "    skipping child space <" << tSpace->GetName() << "> because distance is greater than trajectory radius" << eom );
                    continue;
                }

                // examine intersection function
                fIntermediateParticle.SetCurrentSpace( tSpace );

                // calculate initial intersection
                tInitialIntersection = ( tInitialPoint - tSpace->Point( tInitialPoint )).Dot( tSpace->Normal( tInitialPoint ) );
                navmsg_debug( "    initial intersection to child space <" << tSpace->GetName() << "> is <" << tInitialIntersection << ">" << eom );

                // calculate final intersection
                tFinalIntersection = ( tFinalPoint - tSpace->Point( tFinalPoint )).Dot( tSpace->Normal( tFinalPoint ) );
                navmsg_debug( "    final intersection to child space <" << tSpace->GetName() << "> is <" << tFinalIntersection << ">" << eom );

                // if the initial intersection function is within tolerance, skip the child space
                if( fabs( tInitialIntersection / aTrajectoryRadius ) < fTolerance )
                {
                    navmsg_debug( "    skipping child space <" << tSpace->GetName() << "> because intersection function is within tolerance" << eom );
                    continue;
                }

                // if the intersection function signs are the same, skip the child space
                if( (tInitialIntersection < 0.) == (tFinalIntersection < 0.) )
                {
                    navmsg_debug( "    skipping child space <" << tSpace->GetName() << "> because intersection function signs are the same" << eom );
                    continue;
                }

                // calculate intersection time
                fIntermediateParticle.SetCurrentSpace( tSpace );
                fSolver.Solve( KMathBracketingSolver::eBrent, this, &KSNavSpace::SpaceIntersectionFunction, 0., 0., aTrajectoryStep, tTime );
                navmsg_debug( "    time to child space <" << tSpace->GetName() << "> is <" << tTime << ">" << eom );

                // if the intersection time is not the smallest, skip the child space
                if( tTime > tSpaceTime )
                {
                    navmsg_debug( "    skipping child space <" << tSpace->GetName() << "> because intersection time is not smallest" << eom );
                    continue;
                }

                // calculate intersection distance
                aTrajectory.ExecuteTrajectory( tTime, fIntermediateParticle );
                tDistance = (fIntermediateParticle.GetPosition() - tSpace->Point( fIntermediateParticle.GetPosition() )).Magnitude();
                navmsg_debug( "    distance to child space <" << tSpace->GetName() << "> is <" << tDistance << ">" << eom );

                // if the intersection distance is outside of tolerance, skip the child space
                if( (tDistance / aTrajectoryRadius) > fTolerance )
                {
                    navmsg_debug( "    skipping child space <" << tSpace->GetName() << "> because intersection distance is outside of tolerance" << eom );
                    continue;
                }

                tSpaceFlag = true;
                tSpaceTime = tTime;
                fParentSpace = NULL;
                fChildSpace = tSpace;
            }

            fChildSpaceRecalculate = false;
            navmsg_debug( "  minimum distance to child spaces is <" << fChildSpaceDistance << ">" << eom );
        }

        //************
        //parent sides
        //************

        if( fParentSideRecalculate == false )
        {
            double tExcursion = (aTrajectoryCenter - fParentSideAnchor).Magnitude() + aTrajectoryRadius;
            if( tExcursion > fParentSideDistance )
            {
                navmsg_debug( "  excursion from parent side anchor exceeds cached distance <" << fParentSideDistance << ">" << eom );

                fParentSideRecalculate = true;
            }
        }
        if( fParentSideRecalculate == true )
        {
            fParentSideAnchor = aTrajectoryCenter;
            fParentSideDistance = numeric_limits< double >::max();
            navmsg_debug( "  minimum distance to parent sides must be recalculated" << eom );

            for( int tParentSideIndex = 0; tParentSideIndex < tCurrentSpace->GetSideCount(); tParentSideIndex++ )
            {
                tSide = tCurrentSpace->GetSide( tParentSideIndex );

                // calculate the distance between the anchor and the parent side
                tDistance = (fParentSideAnchor - tSide->Point( fParentSideAnchor )).Magnitude();
                navmsg_debug( "    distance to parent side <" << tSide->GetName() << "> is <" << tDistance << ">" << eom );

                // if the current distance is less than the current minimum distance replace the current minimum distance with the current distance
                if( tDistance < fParentSideDistance )
                {
                    fParentSideDistance = tDistance;
                }

                // if this distance is greater than the trajectory radius, skip the parent side
                if( tDistance > aTrajectoryRadius )
                {
                    navmsg_debug( "    skipping parent side <" << tSide->GetName() << "> because distance is greater than trajectory radius" << eom );
                    continue;
                }

                // examine intersection function
                fIntermediateParticle.SetCurrentSide( tSide );

                // calculate initial intersection
                tInitialIntersection = ( tInitialPoint - tSide->Point( tInitialPoint )).Dot( tSide->Normal( tInitialPoint ) );
                navmsg_debug( "    initial intersection to parent side <" << tSide->GetName() << "> is <" << tInitialIntersection << ">" << eom );

                // calculate final intersection
                tFinalIntersection = ( tFinalPoint - tSide->Point( tFinalPoint )).Dot( tSide->Normal( tFinalPoint ) );
                navmsg_debug( "    final intersection to parent side <" << tSide->GetName() << "> is <" << tFinalIntersection << ">" << eom );

                // if the initial intersection function is within tolerance, skip the parent side
                if( fabs( tInitialIntersection / aTrajectoryRadius ) < fTolerance )
                {
                    navmsg_debug( "    skipping parent side <" << tSide->GetName() << "> because intersection function is within tolerance" << eom );
                    continue;
                }

                // if the intersection function signs are the same, skip the parent side
                if( (tInitialIntersection < 0.) == (tFinalIntersection < 0.) )
                {
                    navmsg_debug( "    skipping parent side <" << tSide->GetName() << "> because intersection function signs are the same" << eom );
                    continue;
                }

                // calculate intersection time
                fIntermediateParticle.SetCurrentSide( tSide );
                fSolver.Solve( KMathBracketingSolver::eBrent, this, &KSNavSpace::SideIntersectionFunction, 0., 0., aTrajectoryStep, tTime );
                navmsg_debug( "    time to parent side <" << tSide->GetName() << "> is <" << tTime << ">" << eom );

                // if the intersection time is not the smallest, skip the parent side
                if( tTime > tSideTime )
                {
                    navmsg_debug( "    skipping parent side <" << tSide->GetName() << "> because intersection time is not smallest" << eom );
                    continue;
                }

                // calculate intersection distance
                aTrajectory.ExecuteTrajectory( tTime, fIntermediateParticle );
                tDistance = (fIntermediateParticle.GetPosition() - tSide->Point( fIntermediateParticle.GetPosition() )).Magnitude();
                navmsg_debug( "    distance to parent side <" << tSide->GetName() << "> is <" << tDistance << ">" << eom );

                // if the intersection distance is outside of tolerance, skip the parent side
                if( (tDistance / aTrajectoryRadius) > fTolerance )
                {
                    navmsg_debug( "    skipping parent side <" << tSide->GetName() << "> because intersection distance is outside of tolerance" << eom );
                    continue;
                }

                tSideFlag = true;
                tSideTime = tTime;
                fParentSide = tSide;
                fChildSide = NULL;
            }

            fParentSideRecalculate = false;
            navmsg_debug( "  minimum distance to parent sides is <" << fParentSideDistance << ">" << eom );
        }

        //***********
        //child sides
        //***********

        if( fChildSideRecalculate == false )
        {
            double tExcursion = (aTrajectoryCenter - fChildSideAnchor).Magnitude() + aTrajectoryRadius;
            if( tExcursion > fChildSideDistance )
            {
                navmsg_debug( "  excursion from child side anchor exceeds cached distance <" << fChildSideDistance << ">" << eom );

                fChildSideRecalculate = true;
            }
        }
        if( fChildSideRecalculate == true )
        {
            fChildSideAnchor = aTrajectoryCenter;
            fChildSideDistance = numeric_limits< double >::max();
            navmsg_debug( "  minimum distance to child sides must be recalculated" << eom );

            for( int tSpaceIndex = 0; tSpaceIndex < tCurrentSpace->GetSpaceCount(); tSpaceIndex++ )
            {
                tSpace = tCurrentSpace->GetSpace( tSpaceIndex );
                for( int tInternalSideIndex = 0; tInternalSideIndex < tSpace->GetSideCount(); tInternalSideIndex++ )
                {
                    tSide = tSpace->GetSide( tInternalSideIndex );

                    // calculate the distance between the anchor and the child side
                    tDistance = (fChildSideAnchor - tSide->Point( fChildSideAnchor )).Magnitude();
                    navmsg_debug( "    distance to child side <" << tSide->GetName() << "> is <" << tDistance << ">" << eom );

                    // if the current distance is less than the current minimum distance replace the current minimum distance with the current distance
                    if( tDistance < fChildSideDistance )
                    {
                        fChildSideDistance = tDistance;
                    }

                    // if this distance is greater than the trajectory radius, skip the child side
                    if( tDistance > aTrajectoryRadius )
                    {
                        navmsg_debug( "    skipping child side <" << tSide->GetName() << "> because distance is greater than trajectory radius" << eom );
                        continue;
                    }

                    // examine intersection function
                    fIntermediateParticle.SetCurrentSide( tSide );

                    // calculate initial intersection
                    tInitialIntersection = ( tInitialPoint - tSide->Point( tInitialPoint )).Dot( tSide->Normal( tInitialPoint ) );
                    navmsg_debug( "    initial intersection to child side <" << tSide->GetName() << "> is <" << tInitialIntersection << ">" << eom );

                    // calculate final intersection
                    tFinalIntersection = ( tFinalPoint - tSide->Point( tFinalPoint )).Dot( tSide->Normal( tFinalPoint ) );
                    navmsg_debug( "    final intersection to child side <" << tSide->GetName() << "> is <" << tFinalIntersection << ">" << eom );

                    // if the initial intersection function is within tolerance, skip the child side
                    if( fabs( tInitialIntersection / aTrajectoryRadius ) < fTolerance )
                    {
                        navmsg_debug( "    skipping child side <" << tSide->GetName() << "> because intersection function is within tolerance" << eom );
                        continue;
                    }

                    // if the intersection function signs are the same, skip the child side
                    if( (tInitialIntersection < 0.) == (tFinalIntersection < 0.) )
                    {
                        navmsg_debug( "    skipping child side <" << tSide->GetName() << "> because intersection function signs are the same" << eom );
                        continue;
                    }

                    // calculate intersection time
                    fIntermediateParticle.SetCurrentSide( tSide );
                    fSolver.Solve( KMathBracketingSolver::eBrent, this, &KSNavSpace::SideIntersectionFunction, 0., 0., aTrajectoryStep, tTime );
                    navmsg_debug( "    time to child side <" << tSide->GetName() << "> is <" << tTime << ">" << eom );

                    // if the intersection time is not the smallest, skip the child side
                    if( tTime > tSideTime )
                    {
                        navmsg_debug( "    skipping child side <" << tSide->GetName() << "> because intersection time is not smallest" << eom );
                        continue;
                    }

                    // calculate intersection distance
                    aTrajectory.ExecuteTrajectory( tTime, fIntermediateParticle );
                    tDistance = (fIntermediateParticle.GetPosition() - tSide->Point( fIntermediateParticle.GetPosition() )).Magnitude();
                    navmsg_debug( "    distance to child side <" << tSide->GetName() << "> is <" << tDistance << ">" << eom );

                    // if the intersection distance is outside of tolerance, skip the child side
                    if( (tDistance / aTrajectoryRadius) > fTolerance )
                    {
                        navmsg_debug( "    skipping child side <" << tSide->GetName() << "> because intersection distance is outside of tolerance" << eom );
                        continue;
                    }

                    tSideFlag = true;
                    tSideTime = tTime;
                    fParentSide = NULL;
                    fChildSide = tSide;
                }
            }

            fChildSideRecalculate = false;
            navmsg_debug( "  minimum distance to child sides is <" << fChildSideDistance << ">" << eom );
        }

        //**************
        //child surfaces
        //**************

        if( fChildSurfaceRecalculate == false )
        {
            double tExcursion = (aTrajectoryCenter - fChildSurfaceAnchor).Magnitude() + aTrajectoryRadius;
            if( tExcursion > fChildSurfaceDistance )
            {
                fChildSurfaceRecalculate = true;

                navmsg_debug( "  excursion from child surface anchor exceeds cached distance <" << fChildSurfaceDistance << ">" << eom );
            }
        }
        if( fChildSurfaceRecalculate == true )
        {
            fChildSurfaceAnchor = aTrajectoryCenter;
            fChildSurfaceDistance = numeric_limits< double >::max();
            navmsg_debug( "  minimum distance to child surfaces must be recalculated" << eom );

            for( int tSurfaceIndex = 0; tSurfaceIndex < tCurrentSpace->GetSurfaceCount(); tSurfaceIndex++ )
            {
                tSurface = tCurrentSpace->GetSurface( tSurfaceIndex );

                // calculate the distance between the anchor and the child surface
                tDistance = (fChildSurfaceAnchor - tSurface->Point( fChildSurfaceAnchor )).Magnitude();
                navmsg_debug( "    distance to child surface <" << tSurface->GetName() << "> is <" << tDistance << ">" << eom );

                // if the current distance is less than the current minimum distance replace the current minimum distance with the current distance
                if( tDistance < fChildSurfaceDistance )
                {
                    fChildSurfaceDistance = tDistance;
                }

                // if this distance is greater than the trajectory radius, skip the child surface
                if( tDistance > aTrajectoryRadius )
                {
                    navmsg_debug( "    skipping child surface <" << tSurface->GetName() << "> because distance is greater than trajectory radius" << eom );
                    continue;
                }

                // examine intersection function
                fIntermediateParticle.SetCurrentSurface( tSurface );

                // calculate initial intersection
                tInitialIntersection = ( tInitialPoint - tSurface->Point( tInitialPoint )).Dot( tSurface->Normal( tInitialPoint ) );
                navmsg_debug( "    initial intersection to child surface <" << tSurface->GetName() << "> is <" << tInitialIntersection << ">" << eom );

                // calculate final intersection
                tFinalIntersection = ( tFinalPoint - tSurface->Point( tFinalPoint )).Dot( tSurface->Normal( tFinalPoint ) );
                navmsg_debug( "    final intersection to child surface <" << tSurface->GetName() << "> is <" << tFinalIntersection << ">" << eom );

                // if the initial intersection function is within tolerance, skip the child surface
                if( fabs( tInitialIntersection / aTrajectoryRadius ) < fTolerance )
                {
                    navmsg_debug( "    skipping child surface <" << tSurface->GetName() << "> because intersection function is within tolerance" << eom );
                    continue;
                }

                // if the intersection function signs are the same, skip the child surface
                if( (tInitialIntersection < 0.) == (tFinalIntersection < 0.) )
                {
                    navmsg_debug( "    skipping child surface <" << tSurface->GetName() << "> because intersection function signs are the same" << eom );
                    continue;
                }

                // calculate intersection time
                fIntermediateParticle.SetCurrentSurface( tSurface );
                fSolver.Solve( KMathBracketingSolver::eBrent, this, &KSNavSpace::SurfaceIntersectionFunction, 0., 0., aTrajectoryStep, tTime );
                navmsg_debug( "    time to child surface <" << tSurface->GetName() << "> is <" << tTime << ">" << eom );

                // if the intersection time is not the smallest, skip the child surface
                if( tTime > tSurfaceTime )
                {
                    navmsg_debug( "    skipping child surface <" << tSurface->GetName() << "> because intersection time is not smallest" << eom );
                    continue;
                }

                // calculate intersection distance
                aTrajectory.ExecuteTrajectory( tTime, fIntermediateParticle );
                tDistance = (fIntermediateParticle.GetPosition() - tSurface->Point( fIntermediateParticle.GetPosition() )).Magnitude();
                navmsg_debug( "    distance to child surface <" << tSurface->GetName() << "> is <" << tDistance << ">" << eom );

                // if the intersection distance is outside of tolerance, skip the child surface
                if( (tDistance / aTrajectoryRadius) > fTolerance )
                {
                    navmsg_debug( "    skipping child surface <" << tSurface->GetName() << "> because intersection distance is outside of tolerance" << eom );
                    continue;
                }

                tSurfaceFlag = true;
                tSurfaceTime = tTime;
                fChildSurface = tSurface;
            }

            fChildSurfaceRecalculate = false;
            navmsg_debug( "  minimum distance to child surfaces is <" << fChildSurfaceDistance << ">" << eom );
        }

        //******************
        //case determination
        //******************

        if( tSideFlag == true )
        {
            if( tSideTime < tSurfaceTime )
            {
                fChildSurface = NULL;
                fParentSpace = NULL;
                fChildSpace = NULL;

                aTrajectory.ExecuteTrajectory( tSideTime, aNavigationParticle );
                aNavigationStep = tSideTime;
                aNavigationFlag = true;

                navmsg_debug( "  particle may cross side" << eom );

                return;
            }
        }

        if( (tSurfaceFlag == true) || (tSpaceFlag == true) )
        {
            if( tSurfaceTime < tSpaceTime )
            {
                fParentSpace = NULL;
                fChildSpace = NULL;

                aTrajectory.ExecuteTrajectory( tSurfaceTime, aNavigationParticle );
                aNavigationStep = tSurfaceTime;
                aNavigationFlag = true;

                navmsg_debug( "  particle may cross surface" << eom );

                return;
            }
            else
            {
                fChildSurface = NULL;

                aTrajectory.ExecuteTrajectory( tSpaceTime, aNavigationParticle );
                aNavigationStep = tSpaceTime;
                aNavigationFlag = true;

                navmsg_debug( "  particle may change space" << eom );

                return;
            }
        }

        aNavigationParticle = aTrajectoryFinalParticle;
        aNavigationStep = aTrajectoryStep;
        aNavigationFlag = false;

        navmsg_debug( "  no navigation occurred" << eom );

        return;
    }
    void KSNavSpace::ExecuteNavigation( const KSParticle& aNavigationParticle, KSParticle& aFinalParticle, KSParticleQueue& aParticleQueue ) const
    {
        navmsg_debug( "navigation space <" << this->GetName() << "> executing navigation:" << eom );

        if( fParentSpace != NULL )
        {
            navmsg_debug( "  parent space <" << fParentSpace->GetName() << "> was exited" << eom );

            aFinalParticle = aNavigationParticle;
            aFinalParticle.SetCurrentSpace( fParentSpace->GetParent() );
            aFinalParticle.SetCurrentSurface( NULL );
            aFinalParticle.SetLabel( GetName() );
            aFinalParticle.AddLabel( fParentSpace->GetName() );
            aFinalParticle.AddLabel( "exit" );
            fParentSpace->Exit();

            if( fExitSplit == true )
            {
                KSParticle* tExitSplitParticle = new KSParticle( aFinalParticle );
                tExitSplitParticle->SetLabel( GetName() );
                tExitSplitParticle->AddLabel( fParentSpace->GetName() );
                tExitSplitParticle->AddLabel( "exit" );
                aParticleQueue.push_back( tExitSplitParticle );
                aFinalParticle.SetActive( false );
            }

            fParentSpace = NULL;
            return;
        }

        if( fChildSpace != NULL )
        {
            navmsg_debug( "  child space <" << fChildSpace->GetName() << "> was entered" << eom );

            aFinalParticle = aNavigationParticle;
            aFinalParticle.SetCurrentSpace( fChildSpace );
            aFinalParticle.SetCurrentSurface( NULL );
            aFinalParticle.SetLabel( GetName() );
            aFinalParticle.AddLabel( fChildSpace->GetName() );
            aFinalParticle.AddLabel( "enter" );
            fChildSpace->Enter();

            if( fEnterSplit == true )
            {
                KSParticle* tEnterSplitParticle = new KSParticle( aFinalParticle );
                tEnterSplitParticle->SetLabel( GetName() );
                tEnterSplitParticle->AddLabel( fChildSpace->GetName() );
                tEnterSplitParticle->AddLabel( "enter" );
                aParticleQueue.push_back( tEnterSplitParticle );
                aFinalParticle.SetActive( false );
            }

            fChildSpace = NULL;
            return;
        }

        if( fParentSide != NULL )
        {
            navmsg_debug( "  parent side <" << fParentSide->GetName() << "> was crossed" << eom );

            aFinalParticle = aNavigationParticle;
            aFinalParticle.SetCurrentSide( fParentSide );
            aFinalParticle.SetLabel( GetName() );
            aFinalParticle.AddLabel( fParentSide->GetName() );
            aFinalParticle.AddLabel( "crossed" );
            fParentSide->On();

            fParentSide = NULL;
            return;
        }

        if( fChildSide != NULL )
        {
            navmsg_debug( "  child side <" << fChildSide->GetName() << "> was crossed" << eom );

            aFinalParticle = aNavigationParticle;
            aFinalParticle.SetCurrentSide( fChildSide );
            aFinalParticle.SetLabel( GetName() );
            aFinalParticle.AddLabel( fChildSide->GetName() );
            aFinalParticle.AddLabel( "crossed" );
            fChildSide->On();

            fChildSide = NULL;
            return;
        }

        if( fChildSurface != NULL )
        {
            navmsg_debug( "  child surface <" << fChildSurface->GetName() << "> was crossed" << eom );

            aFinalParticle = aNavigationParticle;
            aFinalParticle.SetCurrentSurface( fChildSurface );
            aFinalParticle.SetLabel( GetName() );
            aFinalParticle.AddLabel( fChildSurface->GetName() );
            aFinalParticle.AddLabel( "crossed" );
            fChildSurface->On();

            fChildSurface = NULL;
            return;
        }

        navmsg( eError ) << "could not determine space navigation" << eom;
        return;
    }

    void KSNavSpace::StartNavigation( KSParticle& aParticle, KSSpace* aRoot )
    {
        navmsg_debug( "navigation space <" << this->GetName() << "> starting navigation:" << eom );

        // reset navigation
        fParentSideRecalculate = true;
        fChildSideRecalculate = true;
        fChildSurfaceRecalculate = true;
        fParentSpaceRecalculate = true;
        fChildSpaceRecalculate = true;

        if( aParticle.GetCurrentSpace() == NULL )
        {
            navmsg_debug( "  computing fresh initial state" << eom );

            int tIndex = 0;
            KSSpace* tParentSpace = aRoot;
            KSSpace* tSpace = NULL;
            while( tIndex < tParentSpace->GetSpaceCount() )
            {
                tSpace = tParentSpace->GetSpace( tIndex );
                if( tSpace->Outside( aParticle.GetPosition() ) == false )
                {
                    navmsg_debug( "  activating space <" << tSpace->GetName() << ">" << eom );

                    tSpace->Enter();
                    tParentSpace = tSpace;
                    tIndex = 0;
                }
                else
                {
                    navmsg_debug( "  skipping space <" << tSpace->GetName() << ">" << eom );

                    tIndex++;
                }
            }

            aParticle.SetCurrentSpace( tParentSpace );
        }
        else
        {
            navmsg_debug( "  entering given initial state" << eom );

            KSSpace* tSpace = aParticle.GetCurrentSpace();
            KSSurface* tSurface = aParticle.GetCurrentSurface();
            KSSide* tSide = aParticle.GetCurrentSide();
            deque< KSSpace* > tSequence;

            // get into the correct space state
            while( tSpace != aRoot )
            {
                tSequence.push_front( tSpace );
                tSpace = tSpace->GetParent();
            }
            for( deque< KSSpace* >::iterator tIt = tSequence.begin(); tIt != tSequence.end(); tIt++ )
            {
                tSpace = *tIt;

                navmsg_debug( "  entering space <" << tSpace->GetName() << ">" << eom );

                tSpace->Enter();
            }

            // get into the correct surface state
            if( tSurface != NULL )
            {
                navmsg_debug( "  child surface was crossed" << eom );

                aParticle.SetCurrentSide( NULL );
                aParticle.SetCurrentSurface( NULL );
                aParticle.SetCurrentSpace( tSpace );

                return;
            }

            // get into the correct side state
            if( tSide != NULL )
            {
                KThreeVector tMomentum = aParticle.GetMomentum();
                KThreeVector tNormal = aParticle.GetCurrentSide()->Normal( aParticle.GetPosition() );

                if( tSpace == tSide->GetInsideParent() )
                {
                    if( tMomentum.Dot( tNormal ) > 0. )
                    {
                        navmsg_debug( "  transmission occurred on boundary <" << tSide->GetName() << "> of parent space <" << tSide->GetInsideParent()->GetName() << ">" << eom );

                        aParticle.SetCurrentSide( NULL );
                        aParticle.SetCurrentSurface( NULL );
                        aParticle.SetCurrentSpace( tSide->GetOutsideParent() );
                        tSide->GetInsideParent()->Exit();

                        return;
                    }
                    else
                    {
                        navmsg_debug( "  reflection occurred on boundary <" << tSide->GetName() << "> of parent space <" << tSide->GetInsideParent()->GetName() << ">" << eom );

                        aParticle.SetCurrentSide( NULL );
                        aParticle.SetCurrentSurface( NULL );
                        aParticle.SetCurrentSpace( tSide->GetInsideParent() );

                        return;
                    }
                }

                if( tSpace == tSide->GetOutsideParent() )
                {
                    if( tMomentum.Dot( tNormal ) < 0. )
                    {
                        navmsg_debug( "  transmission occurred on boundary <" << tSide->GetName() << "> of child space <" << tSide->GetInsideParent()->GetName() << ">" << eom );

                        aParticle.SetCurrentSide( NULL );
                        aParticle.SetCurrentSurface( NULL );
                        aParticle.SetCurrentSpace( tSide->GetInsideParent() );
                        tSide->GetInsideParent()->Enter();

                        return;
                    }
                    else
                    {
                        navmsg_debug( "  reflection occurred on boundary <" << tSide->GetName() << "> of child space <" << tSide->GetInsideParent()->GetName() << ">" << eom );

                        aParticle.SetCurrentSide( NULL );
                        aParticle.SetCurrentSurface( NULL );
                        aParticle.SetCurrentSpace( tSide->GetOutsideParent() );

                        return;
                    }
                }

            }
        }

        return;
    }
    void KSNavSpace::StopNavigation( KSParticle& aParticle, KSSpace* aRoot )
    {
        // reset navigation
        fParentSideRecalculate = true;
        fChildSideRecalculate = true;
        fChildSurfaceRecalculate = true;
        fParentSpaceRecalculate = true;
        fChildSpaceRecalculate = true;

        deque< KSSpace* > tSpaces;
        KSSpace* tSpace = aParticle.GetCurrentSpace();
        KSSurface* tSurface = aParticle.GetCurrentSurface();
        KSSide* tSide = aParticle.GetCurrentSide();

        // undo side state
        if( tSide != NULL )
        {
            navmsg_debug( "  deactivating side <" << tSide->GetName() << ">" << eom );

            tSide->Off();
        }

        // undo surface state
        if( tSurface != NULL )
        {
            navmsg_debug( "  deactivating surface <" << tSurface->GetName() << ">" << eom );

            tSurface->Off();
        }

        // undo space state
        while( tSpace != aRoot )
        {
            tSpaces.push_back( tSpace );
            tSpace = tSpace->GetParent();
        }
        for( deque< KSSpace* >::iterator tIt = tSpaces.begin(); tIt != tSpaces.end(); tIt++ )
        {
            tSpace = *tIt;

            navmsg_debug( "  deactivating space <" << tSpace->GetName() << ">" << eom );

            tSpace->Exit();
        }

        return;
    }

    void KSNavSpace::ActivateComponent()
    {
        fCurrentSpace = NULL;
        fParentSideRecalculate = true;
        fChildSideRecalculate = true;
        fChildSurfaceRecalculate = true;
        fParentSpaceRecalculate = true;
        fChildSpaceRecalculate = true;
        return;
    }
    void KSNavSpace::DeactivateComponent()
    {
        fCurrentSpace = NULL;
        fParentSideRecalculate = true;
        fChildSideRecalculate = true;
        fChildSurfaceRecalculate = true;
        fParentSpaceRecalculate = true;
        fChildSpaceRecalculate = true;
        return;
    }

    double KSNavSpace::SpaceIntersectionFunction( const double& aTime )
    {
        fCurrentTrajectory->ExecuteTrajectory( aTime, fIntermediateParticle );
        KThreeVector tParticlePoint = fIntermediateParticle.GetPosition();
        KThreeVector tSpacePoint = fIntermediateParticle.GetCurrentSpace()->Point( tParticlePoint );
        KThreeVector tSpaceNormal = fIntermediateParticle.GetCurrentSpace()->Normal( tParticlePoint );
        return (tParticlePoint - tSpacePoint).Dot( tSpaceNormal );
    }
    double KSNavSpace::SurfaceIntersectionFunction( const double& aTime )
    {
        fCurrentTrajectory->ExecuteTrajectory( aTime, fIntermediateParticle );
        KThreeVector tParticlePoint = fIntermediateParticle.GetPosition();
        KThreeVector tSurfacePoint = fIntermediateParticle.GetCurrentSurface()->Point( tParticlePoint );
        KThreeVector tSurfaceNormal = fIntermediateParticle.GetCurrentSurface()->Normal( tParticlePoint );
        return (tParticlePoint - tSurfacePoint).Dot( tSurfaceNormal );
    }
    double KSNavSpace::SideIntersectionFunction( const double& aTime )
    {
        fCurrentTrajectory->ExecuteTrajectory( aTime, fIntermediateParticle );
        KThreeVector tParticlePoint = fIntermediateParticle.GetPosition();
        KThreeVector tSidePoint = fIntermediateParticle.GetCurrentSide()->Point( tParticlePoint );
        KThreeVector tSideNormal = fIntermediateParticle.GetCurrentSide()->Normal( tParticlePoint );
        return (tParticlePoint - tSidePoint).Dot( tSideNormal );
    }

}
