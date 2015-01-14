#include "KSTrajTrajectoryAdiabatic.h"
#include "KSTrajectoriesMessage.h"

#include "KConst.h"

#include <limits>
using std::numeric_limits;

namespace Kassiopeia
{

    KSTrajTrajectoryAdiabatic::KSTrajTrajectoryAdiabatic() :
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
    KSTrajTrajectoryAdiabatic::KSTrajTrajectoryAdiabatic( const KSTrajTrajectoryAdiabatic& aCopy ) :
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
    KSTrajTrajectoryAdiabatic* KSTrajTrajectoryAdiabatic::Clone() const
    {
        return new KSTrajTrajectoryAdiabatic( *this );
    }
    KSTrajTrajectoryAdiabatic::~KSTrajTrajectoryAdiabatic()
    {
    }

    void KSTrajTrajectoryAdiabatic::SetIntegrator( KSTrajAdiabaticIntegrator* anIntegrator )
    {
        if( fIntegrator == NULL )
        {
            fIntegrator = anIntegrator;
            return;
        }
        trajmsg( eError ) << "cannot set integrator in <" << this->GetName() << "> with <" << anIntegrator << ">" << eom;
        return;
    }
    void KSTrajTrajectoryAdiabatic::ClearIntegrator( KSTrajAdiabaticIntegrator* anIntegrator )
    {
        if( fIntegrator == anIntegrator )
        {
            fIntegrator = NULL;
            return;
        }
        trajmsg( eError ) << "cannot clear integrator in <" << this->GetName() << "> with <" << anIntegrator << ">" << eom;
        return;
    }

    void KSTrajTrajectoryAdiabatic::SetInterpolator( KSTrajAdiabaticInterpolator* anInterpolator )
    {
        if( fInterpolator == NULL )
        {
            fInterpolator = anInterpolator;
            return;
        }
        trajmsg( eError ) << "cannot set interpolator in <" << this->GetName() << "> with <" << anInterpolator << ">" << eom;
        return;
    }
    void KSTrajTrajectoryAdiabatic::ClearInterpolator( KSTrajAdiabaticInterpolator* anInterpolator )
    {
        if( fInterpolator == anInterpolator )
        {
            fInterpolator = NULL;
            return;
        }
        trajmsg( eError ) << "cannot clear interpolator in <" << this->GetName() << "> with <" << anInterpolator << ">" << eom;
        return;
    }

    void KSTrajTrajectoryAdiabatic::AddTerm( KSTrajAdiabaticDifferentiator* aTerm )
    {
        if( fTerms.AddElement( aTerm ) != -1 )
        {
            return;
        }
        trajmsg( eError ) << "cannot add term <" << aTerm << "> to <" << this->GetName() << ">" << eom;
        return;
    }
    void KSTrajTrajectoryAdiabatic::RemoveTerm( KSTrajAdiabaticDifferentiator* aTerm )
    {
        if( fTerms.RemoveElement( aTerm ) != -1 )
        {
            return;
        }
        trajmsg( eError ) << "cannot remove term <" << aTerm << "> from <" << this->GetName() << ">" << eom;
        return;
    }

    void KSTrajTrajectoryAdiabatic::AddControl( KSTrajAdiabaticControl* aControl )
    {
        if( fControls.AddElement( aControl ) != -1 )
        {
            return;
        }
        trajmsg( eError ) << "cannot add step <" << aControl << "> to <" << this->GetName() << ">" << eom;
        return;
    }
    void KSTrajTrajectoryAdiabatic::RemoveControl( KSTrajAdiabaticControl* aControl )
    {
        if( fControls.RemoveElement( aControl ) != -1 )
        {
            return;
        }
        trajmsg( eError ) << "cannot remove step <" << aControl << "> from <" << this->GetName() << ">" << eom;
        return;
    }

    void KSTrajTrajectoryAdiabatic::CalculateTrajectory( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KThreeVector& aCenter, double& aRadius, double& aTimeStep )
    {
        fInitialParticle = fFinalParticle;
        fInitialParticle.PullFrom( anInitialParticle );

        //spray
        trajmsg_debug( "initial real position: " << fInitialParticle.GetPosition() << ret )
        trajmsg_debug( "initial real momentum: " << fInitialParticle.GetMomentum() << ret )
        trajmsg_debug( "initial gc position: " << fInitialParticle.GetGuidingCenter() << ret )
        trajmsg_debug( "initial gc alpha: " << fInitialParticle.GetAlpha() << ret )
        trajmsg_debug( "initial gc beta: " << fInitialParticle.GetBeta() << ret )
        trajmsg_debug( "initial parallel momentum: <" << fInitialParticle[5] << ">" << ret )
        trajmsg_debug( "initial perpendicular momentum: <" << fInitialParticle[6] << ">" << ret )
		trajmsg_debug( "initial kinetic energy is: <" << fInitialParticle.GetKineticEnergy()/ KConst::Q() << ">" << eom )

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

            trajmsg_debug( "time step is <" << tSmallestStep << ">" << eom );

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

        //compute rotation minimizing frame via double-reflection
        const KThreeVector tInitialPosition = fInitialParticle.GetGuidingCenter();
        const KThreeVector tInitialTangent = fInitialParticle.GetMagneticField().Unit();
        const KThreeVector tInitialNormal = fInitialParticle.GetAlpha();
        const KThreeVector tFinalPosition = fFinalParticle.GetGuidingCenter();
        const KThreeVector tFinalTangent = fFinalParticle.GetMagneticField().Unit();

        KThreeVector tReflectionOneVector = tFinalPosition - tInitialPosition;
        double tReflectionOne = tReflectionOneVector.MagnitudeSquared();
        KThreeVector tTangentA = tInitialTangent - (2. / tReflectionOne) * (tReflectionOneVector.Dot( tInitialTangent )) * tReflectionOneVector;
        KThreeVector tNormalA = tInitialNormal - (2. / tReflectionOne) * (tReflectionOneVector.Dot( tInitialNormal )) * tReflectionOneVector;

        KThreeVector tReflectionTwoVector = tFinalTangent - tTangentA;
        double tReflectionTwo = tReflectionTwoVector.MagnitudeSquared();
        KThreeVector tNormalB = tNormalA - (2. / tReflectionTwo) * (tReflectionTwoVector.Dot( tNormalA )) * tReflectionTwoVector;
        KThreeVector tNormalC = tNormalB - tNormalB.Dot( tFinalTangent ) * tFinalTangent;
        KThreeVector tFinalNormal = tNormalC.Unit();
        KThreeVector tFinalBinormal = tFinalTangent.Cross( tFinalNormal ).Unit();

        fFinalParticle.SetAlpha( tFinalNormal );
        fFinalParticle.SetBeta( tFinalBinormal );
        fFinalParticle.PushTo( aFinalParticle );
        aFinalParticle.SetLabel( GetName() );

        trajmsg_debug( "final real position: " << fFinalParticle.GetPosition() << ret )
        trajmsg_debug( "final real momentum: " << fFinalParticle.GetMomentum() << ret )
        trajmsg_debug( "final gc position: " << fFinalParticle.GetGuidingCenter() << ret )
        trajmsg_debug( "final gc alpha: " << fFinalParticle.GetAlpha() << ret )
        trajmsg_debug( "final gc beta: " << fFinalParticle.GetBeta() << ret )
        trajmsg_debug( "final parallel momentum: <" << fFinalParticle[5] << ">" << ret )
        trajmsg_debug( "final perpendicular momentum: <" << fFinalParticle[6] << ">" << ret )
		trajmsg_debug( "final kinetic energy is: <" << fFinalParticle.GetKineticEnergy()/ KConst::Q() << ">" << eom )


        KThreeVector tInitialFinalLine = fFinalParticle.GetPosition() - fInitialParticle.GetPosition();
        aCenter = fInitialParticle.GetPosition() + .5 * tInitialFinalLine;
        aRadius = .5 * tInitialFinalLine.Magnitude();
        aTimeStep = tSmallestStep;

        return;
    }

    void KSTrajTrajectoryAdiabatic::ExecuteTrajectory( const double& aTimeStep, KSParticle& anIntermediateParticle ) const
    {
    	if ( fInterpolator )
    	{
			fInterpolator->Interpolate( *this, fInitialParticle, fFinalParticle, aTimeStep, fIntermediateParticle );
			fIntermediateParticle.PushTo( anIntermediateParticle );
			return;
    	}
    	else
    	{
    		trajmsg_debug( "execute trajectory without interpolation: "<<ret )
    		trajmsg_debug( "timestep: "<<aTimeStep <<ret )
            trajmsg_debug( "initial real position: " << fInitialParticle.GetPosition() << ret )
            trajmsg_debug( "initial real momentum: " << fInitialParticle.GetMomentum() << ret )
            trajmsg_debug( "initial gc position: " << fInitialParticle.GetGuidingCenter() << ret )
            trajmsg_debug( "initial gc alpha: " << fInitialParticle.GetAlpha() << ret )
            trajmsg_debug( "initial gc beta: " << fInitialParticle.GetBeta() << ret )
            trajmsg_debug( "initial parallel momentum: <" << fInitialParticle[5] << ">" << ret )
            trajmsg_debug( "initial perpendicular momentum: <" << fInitialParticle[6] << ">" << ret )
    		trajmsg_debug( "initial kinetic energy is: <" << fInitialParticle.GetKineticEnergy()/ KConst::Q() << ">" << eom )

			if ( aTimeStep == 0.0 )
			{
				fIntermediateParticle = fInitialParticle;
				trajmsg_debug( "timestep was 0, using initial particle" << eom )
				fIntermediateParticle.PushTo( anIntermediateParticle );
				return;
			}

            fIntegrator->Integrate( *this, fInitialParticle, aTimeStep, fIntermediateParticle, fError );

            //compute rotation minimizing frame via double-reflection
            const KThreeVector tInitialPosition = fInitialParticle.GetGuidingCenter();
            const KThreeVector tInitialTangent = fInitialParticle.GetMagneticField().Unit();
            const KThreeVector tInitialNormal = fInitialParticle.GetAlpha();
            const KThreeVector tFinalPosition = fIntermediateParticle.GetGuidingCenter();
            const KThreeVector tFinalTangent = fIntermediateParticle.GetMagneticField().Unit();

            KThreeVector tReflectionOneVector = tFinalPosition - tInitialPosition;
            double tReflectionOne = tReflectionOneVector.MagnitudeSquared();
            KThreeVector tTangentA = tInitialTangent - (2. / tReflectionOne) * (tReflectionOneVector.Dot( tInitialTangent )) * tReflectionOneVector;
            KThreeVector tNormalA = tInitialNormal - (2. / tReflectionOne) * (tReflectionOneVector.Dot( tInitialNormal )) * tReflectionOneVector;

            KThreeVector tReflectionTwoVector = tFinalTangent - tTangentA;
            double tReflectionTwo = tReflectionTwoVector.MagnitudeSquared();
            KThreeVector tNormalB = tNormalA - (2. / tReflectionTwo) * (tReflectionTwoVector.Dot( tNormalA )) * tReflectionTwoVector;
            KThreeVector tNormalC = tNormalB - tNormalB.Dot( tFinalTangent ) * tFinalTangent;
            KThreeVector tFinalNormal = tNormalC.Unit();
            KThreeVector tFinalBinormal = tFinalTangent.Cross( tFinalNormal ).Unit();

            fIntermediateParticle.SetAlpha( tFinalNormal );
            fIntermediateParticle.SetBeta( tFinalBinormal );

	        trajmsg_debug( "intermediate real position: " << fIntermediateParticle.GetPosition() << ret )
	        trajmsg_debug( "intermediate real momentum: " << fIntermediateParticle.GetMomentum() << ret )
	        trajmsg_debug( "intermediate gc position: " << fIntermediateParticle.GetGuidingCenter() << ret )
	        trajmsg_debug( "intermediate gc alpha: " << fIntermediateParticle.GetAlpha() << ret )
	        trajmsg_debug( "intermediate gc beta: " << fIntermediateParticle.GetBeta() << ret )
	        trajmsg_debug( "intermediate parallel momentum: <" << fIntermediateParticle[5] << ">" << ret )
	        trajmsg_debug( "intermediate perpendicular momentum: <" << fIntermediateParticle[6] << ">" << ret )
    		trajmsg_debug( "intermediate kinetic energy is: <" << fIntermediateParticle.GetKineticEnergy()/ KConst::Q() << ">" << eom )

            fIntermediateParticle.PushTo( anIntermediateParticle );
	        fFinalParticle = fIntermediateParticle;
    		return;
    	}
    }

    void KSTrajTrajectoryAdiabatic::Differentiate( const KSTrajAdiabaticParticle& aValue, KSTrajAdiabaticDerivative& aDerivative ) const
    {
        double tLongVelocity = aValue.GetLongVelocity();
        double tTransVelocity = aValue.GetTransVelocity();

        aDerivative = 0.;
        aDerivative.AddToTime( 1. );
        aDerivative.AddToSpeed( sqrt( tLongVelocity * tLongVelocity + tTransVelocity * tTransVelocity ) );

        for( int Index = 0; Index < fTerms.End(); Index++ )
        {
            fTerms.ElementAt( Index )->Differentiate( aValue, aDerivative );
        }

        return;
    }

    static const int sKSTrajTrajectoryAdiabaticDict =
        KSDictionary< KSTrajTrajectoryAdiabatic >::AddCommand( &KSTrajTrajectoryAdiabatic::SetIntegrator, &KSTrajTrajectoryAdiabatic::ClearIntegrator, "set_integrator", "clear_integrator" ) +
        KSDictionary< KSTrajTrajectoryAdiabatic >::AddCommand( &KSTrajTrajectoryAdiabatic::SetInterpolator, &KSTrajTrajectoryAdiabatic::ClearInterpolator, "set_interpolator", "clear_interpolator" ) +
        KSDictionary< KSTrajTrajectoryAdiabatic >::AddCommand( &KSTrajTrajectoryAdiabatic::AddTerm, &KSTrajTrajectoryAdiabatic::RemoveTerm, "add_term", "remove_term" ) +
        KSDictionary< KSTrajTrajectoryAdiabatic >::AddCommand( &KSTrajTrajectoryAdiabatic::AddControl, &KSTrajTrajectoryAdiabatic::RemoveControl, "add_control", "remove_control" );

}


