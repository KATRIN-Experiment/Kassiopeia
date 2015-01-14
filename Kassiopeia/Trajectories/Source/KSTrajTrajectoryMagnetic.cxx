#include "KSTrajTrajectoryMagnetic.h"
#include "KSTrajectoriesMessage.h"

#include "KConst.h"

#include <limits>
using std::numeric_limits;

namespace Kassiopeia
{

    KSTrajTrajectoryMagnetic::KSTrajTrajectoryMagnetic() :
            fInitialParticle(),
            fIntermediateParticle(),
            fFinalParticle(),
            fError(),
            fIntegrator( NULL ),
            fInterpolator( NULL ),
            fTerms(),
            fControls(),
            fReverseDirection( false )
    {
    }
    KSTrajTrajectoryMagnetic::KSTrajTrajectoryMagnetic( const KSTrajTrajectoryMagnetic& aCopy ) :
            fInitialParticle( aCopy.fInitialParticle ),
            fIntermediateParticle( aCopy.fIntermediateParticle ),
            fFinalParticle( aCopy.fFinalParticle ),
            fError( aCopy.fError ),
            fIntegrator( aCopy.fIntegrator ),
            fInterpolator( aCopy.fInterpolator ),
            fTerms( aCopy.fTerms ),
            fControls( aCopy.fControls ),
            fReverseDirection( aCopy.fReverseDirection )
    {
    }
    KSTrajTrajectoryMagnetic* KSTrajTrajectoryMagnetic::Clone() const
    {
        return new KSTrajTrajectoryMagnetic( *this );
    }
    KSTrajTrajectoryMagnetic::~KSTrajTrajectoryMagnetic()
    {
    }

    void KSTrajTrajectoryMagnetic::SetIntegrator( KSTrajMagneticIntegrator* anIntegrator )
    {
        if( fIntegrator == NULL )
        {
            fIntegrator = anIntegrator;
            return;
        }
        trajmsg( eError ) << "cannot set integrator in <" << this->GetName() << "> with <" << anIntegrator << ">" << eom;
        return;
    }
    void KSTrajTrajectoryMagnetic::ClearIntegrator( KSTrajMagneticIntegrator* anIntegrator )
    {
        if( fIntegrator == anIntegrator )
        {
            fIntegrator = NULL;
            return;
        }
        trajmsg( eError ) << "cannot clear integrator in <" << this->GetName() << "> with <" << anIntegrator << ">" << eom;
        return;
    }

    void KSTrajTrajectoryMagnetic::SetInterpolator( KSTrajMagneticInterpolator* anInterpolator )
    {
        if( fInterpolator == NULL )
        {
            fInterpolator = anInterpolator;
            return;
        }
        trajmsg( eError ) << "cannot set interpolator in <" << this->GetName() << "> with <" << anInterpolator << ">" << eom;
        return;
    }
    void KSTrajTrajectoryMagnetic::ClearInterpolator( KSTrajMagneticInterpolator* anInterpolator )
    {
        if( fInterpolator == anInterpolator )
        {
            fInterpolator = NULL;
            return;
        }
        trajmsg( eError ) << "cannot clear interpolator in <" << this->GetName() << "> with <" << anInterpolator << ">" << eom;
        return;
    }

    void KSTrajTrajectoryMagnetic::AddTerm( KSTrajMagneticDifferentiator* aTerm )
    {
        if( fTerms.AddElement( aTerm ) != -1 )
        {
            return;
        }
        trajmsg( eError ) << "cannot add term <" << aTerm << "> to <" << this->GetName() << ">" << eom;
        return;
    }
    void KSTrajTrajectoryMagnetic::RemoveTerm( KSTrajMagneticDifferentiator* aTerm )
    {
        if( fTerms.RemoveElement( aTerm ) != -1 )
        {
            return;
        }
        trajmsg( eError ) << "cannot remove term <" << aTerm << "> from <" << this->GetName() << ">" << eom;
        return;
    }

    void KSTrajTrajectoryMagnetic::AddControl( KSTrajMagneticControl* aControl )
    {
        if( fControls.AddElement( aControl ) != -1 )
        {
            return;
        }
        trajmsg( eError ) << "cannot add control <" << aControl << "> to <" << this->GetName() << ">" << eom;
        return;
    }
    void KSTrajTrajectoryMagnetic::RemoveControl( KSTrajMagneticControl* aControl )
    {
        if( fControls.RemoveElement( aControl ) != -1 )
        {
            return;
        }
        trajmsg( eError ) << "cannot remove control <" << aControl << "> from <" << this->GetName() << ">" << eom;
        return;
    }

    void KSTrajTrajectoryMagnetic::SetReverseDirection( const bool& aFlag )
    {
        fReverseDirection = aFlag;
    }
    const bool& KSTrajTrajectoryMagnetic::GetReverseDirection() const
    {
        return fReverseDirection;
    }

    void KSTrajTrajectoryMagnetic::CalculateTrajectory( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KThreeVector& aCenter, double& aRadius, double& aTimeStep )
    {
        fInitialParticle = fFinalParticle;
        fInitialParticle.PullFrom( anInitialParticle );

        //trajmsg_debug( "magnetic trajectory integrating:" << eom );

        bool tFlag;
        double tStep;
        double tSmallestStep = numeric_limits< double >::max();
        for( int tIndex = 0; tIndex < fControls.End(); tIndex++ )
        {
            fControls.ElementAt( tIndex )->Calculate( fInitialParticle, tStep );
            if( tStep < tSmallestStep )
            {
                tSmallestStep = tStep;
            }
        }

        while( true )
        {

            //trajmsg_debug( "  time step: <" << tSmallestStep << ">" << eom );

            fIntegrator->Integrate( *this, fInitialParticle, tSmallestStep, fFinalParticle, fError );

            tFlag = true;
            for( int tIndex = 0; tIndex < fControls.End(); tIndex++ )
            {
                fControls.ElementAt( tIndex )->Check( fInitialParticle, fFinalParticle, fError, tFlag );
                if( tFlag == false )
                {
                    break;
                }
            }

            if( tFlag == true )
            {
                break;
            }
        }

        fFinalParticle.PushTo( aFinalParticle );
        aFinalParticle.SetLabel( GetName() );

        KThreeVector tInitialFinalLine = fFinalParticle.GetPosition() - fInitialParticle.GetPosition();
        aCenter = fInitialParticle.GetPosition() + .5 * tInitialFinalLine;
        aRadius = .5 * tInitialFinalLine.Magnitude();
        aTimeStep = tSmallestStep;

        return;
    }

    void KSTrajTrajectoryMagnetic::ExecuteTrajectory( const double& aTimeStep, KSParticle& anIntermediateParticle ) const
    {
    	if ( fInterpolator != NULL )
    	{
			fInterpolator->Interpolate( *this, fInitialParticle, fFinalParticle, aTimeStep, fIntermediateParticle );
			fIntermediateParticle.PushTo( anIntermediateParticle );
    		return;
    	}
    	else
    	{
            fIntegrator->Integrate( *this, fInitialParticle, aTimeStep, fIntermediateParticle, fError );
			fIntermediateParticle.PushTo( anIntermediateParticle );
    		return;
    	}
    }

    void KSTrajTrajectoryMagnetic::Differentiate( const KSTrajMagneticParticle& aValue, KSTrajMagneticDerivative& aDerivative ) const
    {
        KThreeVector tVelocity = aValue.GetVelocity();

        aDerivative = 0.;
        aDerivative.AddToTime( 1. );
        aDerivative.AddToSpeed( tVelocity.Magnitude() );
        aDerivative.SetDirectionSign( fReverseDirection ? -1 : 1 );

        for( int Index = 0; Index < fTerms.End(); Index++ )
        {
            fTerms.ElementAt( Index )->Differentiate( aValue, aDerivative );
        }

        return;
    }

    static const int sKSTrajTrajectoryMagneticDict =
        KSDictionary< KSTrajTrajectoryMagnetic >::AddCommand( &KSTrajTrajectoryMagnetic::SetIntegrator, &KSTrajTrajectoryMagnetic::ClearIntegrator, "set_integrator", "clear_integrator" ) +
        KSDictionary< KSTrajTrajectoryMagnetic >::AddCommand( &KSTrajTrajectoryMagnetic::SetInterpolator, &KSTrajTrajectoryMagnetic::ClearInterpolator, "set_interpolator", "clear_interpolator" ) +
        KSDictionary< KSTrajTrajectoryMagnetic >::AddCommand( &KSTrajTrajectoryMagnetic::AddTerm, &KSTrajTrajectoryMagnetic::RemoveTerm, "add_term", "remove_term" ) +
        KSDictionary< KSTrajTrajectoryMagnetic >::AddCommand( &KSTrajTrajectoryMagnetic::AddControl, &KSTrajTrajectoryMagnetic::RemoveControl, "add_control", "remove_control" );

}
