#include "KSTrajTrajectoryLinear.h"
#include "KSTrajectoriesMessage.h"

#include "KConst.h"
using katrin::KConst;

namespace Kassiopeia
{

    KSTrajTrajectoryLinear::KSTrajTrajectoryLinear() :
            fLength( 1.e-2 ),
            fTime( 0. ),
            fPosition( 0., 0., 0. ),
            fVelocity( 0., 0., 0. )
    {
    }
    KSTrajTrajectoryLinear::KSTrajTrajectoryLinear( const KSTrajTrajectoryLinear& aCopy ) :
            fLength( aCopy.fLength ),
            fTime( aCopy.fTime ),
            fPosition( aCopy.fPosition ),
            fVelocity( aCopy.fVelocity )
    {
    }
    KSTrajTrajectoryLinear* KSTrajTrajectoryLinear::Clone() const
    {
        return new KSTrajTrajectoryLinear( *this );
    }
    KSTrajTrajectoryLinear::~KSTrajTrajectoryLinear()
    {
    }

    void KSTrajTrajectoryLinear::SetLength( const double& aLength )
    {
        fLength = aLength;
        return;
    }
    const double& KSTrajTrajectoryLinear::GetLength() const
    {
        return fLength;
    }

    void KSTrajTrajectoryLinear::CalculateTrajectory( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KThreeVector& aCenter, double& aRadius, double& aTimeStep )
    {
        fTime = anInitialParticle.GetTime();
        fPosition = anInitialParticle.GetPosition();
        fVelocity = anInitialParticle.GetVelocity();

        aTimeStep = fLength / fVelocity.Magnitude();
        aCenter = fPosition + .5 * aTimeStep * fVelocity;
        aRadius = .5 * aTimeStep * fVelocity.Magnitude();

        aFinalParticle = anInitialParticle;
        aFinalParticle.SetTime( fTime + aTimeStep );
        aFinalParticle.SetPosition( fPosition + aTimeStep * fVelocity );
        aFinalParticle.SetLabel( GetName() );

        return;
    }

    void KSTrajTrajectoryLinear::ExecuteTrajectory( const double& aTimeStep, KSParticle& anIntermediateParticle ) const
    {
        //TODO: it is possible that the intermediate particle needs more initialization than it's receiving here; check this.

        anIntermediateParticle.SetTime( fTime + aTimeStep );
        anIntermediateParticle.SetPosition( fPosition + aTimeStep * fVelocity );

        return;
    }

}



