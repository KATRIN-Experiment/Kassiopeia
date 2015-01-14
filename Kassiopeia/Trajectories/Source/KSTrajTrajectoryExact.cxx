#include "KSTrajTrajectoryExact.h"
#include "KSTrajectoriesMessage.h"

#include "KConst.h"

#include <limits>
using std::numeric_limits;

namespace Kassiopeia
{

    KSTrajTrajectoryExact::KSTrajTrajectoryExact() :
            fInitialParticle(),
            fIntermediateParticle(),
            fFinalParticle(),
            fError(),
            fIntegrator( NULL ),
            fInterpolator( NULL ),
            fTerms(),
            fControls()
    {
    }
    KSTrajTrajectoryExact::KSTrajTrajectoryExact( const KSTrajTrajectoryExact& aCopy ) :
            fInitialParticle( aCopy.fInitialParticle ),
            fIntermediateParticle( aCopy.fIntermediateParticle ),
            fFinalParticle( aCopy.fFinalParticle ),
            fError( aCopy.fError ),
            fIntegrator( aCopy.fIntegrator ),
            fInterpolator( aCopy.fInterpolator ),
            fTerms( aCopy.fTerms ),
            fControls( aCopy.fControls )
    {
    }
    KSTrajTrajectoryExact* KSTrajTrajectoryExact::Clone() const
    {
        return new KSTrajTrajectoryExact( *this );
    }
    KSTrajTrajectoryExact::~KSTrajTrajectoryExact()
    {
    }

    void KSTrajTrajectoryExact::SetIntegrator( KSTrajExactIntegrator* anIntegrator )
    {
        if( fIntegrator == NULL )
        {
            fIntegrator = anIntegrator;
            return;
        }
        trajmsg( eError ) << "cannot set integrator in <" << this->GetName() << "> with <" << anIntegrator << ">" << eom;
        return;
    }
    void KSTrajTrajectoryExact::ClearIntegrator( KSTrajExactIntegrator* anIntegrator )
    {
        if( fIntegrator == anIntegrator )
        {
            fIntegrator = NULL;
            return;
        }
        trajmsg( eError ) << "cannot clear integrator in <" << this->GetName() << "> with <" << anIntegrator << ">" << eom;
        return;
    }

    void KSTrajTrajectoryExact::SetInterpolator( KSTrajExactInterpolator* anInterpolator )
    {
        if( fInterpolator == NULL )
        {
            fInterpolator = anInterpolator;
            return;
        }
        trajmsg( eError ) << "cannot set interpolator in <" << this->GetName() << "> with <" << anInterpolator << ">" << eom;
        return;
    }
    void KSTrajTrajectoryExact::ClearInterpolator( KSTrajExactInterpolator* anInterpolator )
    {
        if( fInterpolator == anInterpolator )
        {
            fInterpolator = NULL;
            return;
        }
        trajmsg( eError ) << "cannot clear interpolator in <" << this->GetName() << "> with <" << anInterpolator << ">" << eom;
        return;
    }

    void KSTrajTrajectoryExact::AddTerm( KSTrajExactDifferentiator* aTerm )
    {
        if( fTerms.AddElement( aTerm ) != -1 )
        {
            return;
        }
        trajmsg( eError ) << "cannot add term <" << aTerm << "> to <" << this->GetName() << ">" << eom;
        return;
    }
    void KSTrajTrajectoryExact::RemoveTerm( KSTrajExactDifferentiator* aTerm )
    {
        if( fTerms.RemoveElement( aTerm ) != -1 )
        {
            return;
        }
        trajmsg( eError ) << "cannot remove term <" << aTerm << "> from <" << this->GetName() << ">" << eom;
        return;
    }

    void KSTrajTrajectoryExact::AddControl( KSTrajExactControl* aControl )
    {
        if( fControls.AddElement( aControl ) != -1 )
        {
            return;
        }
        trajmsg( eError ) << "cannot add control <" << aControl << "> to <" << this->GetName() << ">" << eom;
        return;
    }
    void KSTrajTrajectoryExact::RemoveControl( KSTrajExactControl* aControl )
    {
        if( fControls.RemoveElement( aControl ) != -1 )
        {
            return;
        }
        trajmsg( eError ) << "cannot remove control <" << aControl << "> from <" << this->GetName() << ">" << eom;
        return;
    }

    void KSTrajTrajectoryExact::CalculateTrajectory( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KThreeVector& aCenter, double& aRadius, double& aTimeStep )
    {
        fInitialParticle = fFinalParticle;
        fInitialParticle.PullFrom( anInitialParticle );

        trajmsg_debug( "exact trajectory integrating:" << eom );

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
            trajmsg_debug( "  time step: <" << tSmallestStep << ">" << eom );

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

    void KSTrajTrajectoryExact::ExecuteTrajectory( const double& aTimeStep, KSParticle& anIntermediateParticle ) const
    {
        if( fInterpolator != NULL )
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

    void KSTrajTrajectoryExact::Differentiate( const KSTrajExactParticle& aValue, KSTrajExactDerivative& aDerivative ) const
    {
        KThreeVector tVelocity = aValue.GetVelocity();

        aDerivative = 0.;
        aDerivative.AddToTime( 1. );
        aDerivative.AddToSpeed( tVelocity.Magnitude() );

        for( int Index = 0; Index < fTerms.End(); Index++ )
        {
            fTerms.ElementAt( Index )->Differentiate( aValue, aDerivative );
        }

        return;
    }

    static const int sKSTrajTrajectoryExactDict =
        KSDictionary< KSTrajTrajectoryExact >::AddCommand( &KSTrajTrajectoryExact::SetIntegrator, &KSTrajTrajectoryExact::ClearIntegrator, "set_integrator", "clear_integrator" ) +
        KSDictionary< KSTrajTrajectoryExact >::AddCommand( &KSTrajTrajectoryExact::SetInterpolator, &KSTrajTrajectoryExact::ClearInterpolator, "set_interpolator", "clear_interpolator" ) +
        KSDictionary< KSTrajTrajectoryExact >::AddCommand( &KSTrajTrajectoryExact::AddTerm, &KSTrajTrajectoryExact::RemoveTerm, "add_term", "remove_term" ) +
        KSDictionary< KSTrajTrajectoryExact >::AddCommand( &KSTrajTrajectoryExact::AddControl, &KSTrajTrajectoryExact::RemoveControl, "add_control", "remove_control" );

}
