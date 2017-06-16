#include "KSGenPositionFluxTube.h"
#include "KSGeneratorsMessage.h"
#include "KSParticle.h"
#include "KConst.h"
using katrin::KConst;
#include "KRandom.h"
using katrin::KRandom;

namespace Kassiopeia
{

	KSGenPositionFluxTube::KSGenPositionFluxTube() :
            fPhiValue( NULL ),
            fZValue( NULL ),
            fMagneticFields(),
            fFlux( 0.0191 ),
            fNIntegrationSteps( 1000 ),
            fOnlySurface( true )
    {
    }
	KSGenPositionFluxTube::KSGenPositionFluxTube( const KSGenPositionFluxTube& aCopy ) :
            KSComponent(),
            fPhiValue( aCopy.fPhiValue ),
            fZValue( aCopy.fZValue ),
            fMagneticFields( aCopy.fMagneticFields ),
            fFlux( aCopy.fFlux ),
            fNIntegrationSteps( aCopy.fNIntegrationSteps ),
            fOnlySurface( aCopy.fOnlySurface )
    {
    }
	KSGenPositionFluxTube* KSGenPositionFluxTube::Clone() const
    {
        return new KSGenPositionFluxTube( *this );
    }
	KSGenPositionFluxTube::~KSGenPositionFluxTube()
    {
    }

    void KSGenPositionFluxTube::Dice( KSParticleQueue* aPrimaries )
    {
        if ( !fPhiValue | !fZValue )
            genmsg( eError ) << "phi or z value undefined in composite position creator <" << this->GetName() << ">" << eom;

        KThreeVector tPosition;

        KSParticle* tParticle;
        KSParticleIt tParticleIt;
        KSParticleQueue tParticles;

        double tPhiValue;
        vector< double > tPhiValues;
        vector< double >::iterator tPhiValueIt;

        double tZValue;
        vector< double > tZValues;
        vector< double >::iterator tZValueIt;

        double tRValue;
		double tX;
		double tY;

		double tFlux;
		double tArea;
		double tLastArea;

        fPhiValue->DiceValue( tPhiValues );
        fZValue->DiceValue( tZValues );

		for( tZValueIt = tZValues.begin(); tZValueIt != tZValues.end(); tZValueIt++ )
		{
			tZValue = (*tZValueIt);

			for( tPhiValueIt = tPhiValues.begin(); tPhiValueIt != tPhiValues.end(); tPhiValueIt++ )
			{
				tPhiValue = (KConst::Pi() / 180.) * (*tPhiValueIt);

				tRValue = 0.0;
				tFlux = 0.0;
				tLastArea = 0.0;

				KThreeVector tField;
				//calculate position at z=0 to get approximation for radius
				CalculateField( KThreeVector( 0, 0, tZValue ), 0.0, tField );
				double tRApproximation = sqrt( fFlux / (KConst::Pi() * tField.Magnitude() ) );
				genmsg_debug( "r approximation is <"<<tRApproximation<<">"<<eom);

				//calculate stepsize from 0 to rApproximation
				double tStepSize = tRApproximation / fNIntegrationSteps;

				while( tFlux < fFlux )
				{
					tX = tRValue * cos( tPhiValue );
					tY = tRValue * sin( tPhiValue );
					CalculateField( KThreeVector(tX,tY,tZValue), 0.0, tField );

					tArea = KConst::Pi()*tRValue*tRValue;
					tFlux += tField.Magnitude() * ( tArea - tLastArea);

					genmsg_debug( "r <"<<tRValue<<">"<<eom);
					genmsg_debug( "field "<<tField<<eom);
					genmsg_debug( "area <"<<tArea<<">"<<eom);
					genmsg_debug( "flux <"<<tFlux<<">"<<eom);

					tRValue += tStepSize;
					tLastArea = tArea;
				}

				//correct the last step, to get a tFlux = fFlux
				tRValue = sqrt( tRValue * tRValue - ( tFlux - fFlux ) / ( tField.Magnitude()*KConst::Pi() ) );

				//dice r value if volume option is choosen
				if ( !fOnlySurface )
				{
					tRValue = pow( KRandom::GetInstance().Uniform( 0.0, tRValue*tRValue ), (1./2.) );
				}

				genmsg_debug( "flux tube generator using r of <"<<tRValue<<">"<<eom);

				for( tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++ )
				{
					tParticle = new KSParticle( **tParticleIt );
					tPosition = tRValue * cos( tPhiValue ) * KThreeVector::sXUnit + tRValue * sin( tPhiValue ) * KThreeVector::sYUnit + tZValue * KThreeVector::sZUnit;
					tParticle->SetPosition( tPosition );
					tParticles.push_back( tParticle );
				}

			}

		}

        for( tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++ )
        {
            tParticle = *tParticleIt;
            delete tParticle;
        }

        aPrimaries->assign( tParticles.begin(), tParticles.end() );

        return;
    }


    void KSGenPositionFluxTube::SetPhiValue( KSGenValue* aPhiValue )
    {
        if( fPhiValue == NULL )
        {
            fPhiValue = aPhiValue;
            return;
        }
        genmsg( eError ) << "cannot set phi value <" << aPhiValue->GetName() << "> to composite position cylindrical creator <" << this->GetName() << ">" << eom;
        return;
    }
    void KSGenPositionFluxTube::ClearPhiValue( KSGenValue* anPhiValue )
    {
        if( fPhiValue == anPhiValue )
        {
            fPhiValue = NULL;
            return;
        }
        genmsg( eError ) << "cannot clear phi value <" << anPhiValue->GetName() << "> from composite position cylindrical creator <" << this->GetName() << ">" << eom;
        return;
    }

    void KSGenPositionFluxTube::SetZValue( KSGenValue* anZValue )
    {
        if( fZValue == NULL )
        {
            fZValue = anZValue;
            return;
        }
        genmsg( eError ) << "cannot set z value <" << anZValue->GetName() << "> to composite position cylindrical creator <" << this->GetName() << ">" << eom;
        return;
    }
    void KSGenPositionFluxTube::ClearZValue( KSGenValue* anZValue )
    {
        if( fZValue == anZValue )
        {
            fZValue = NULL;
            return;
        }
        genmsg( eError ) << "cannot clear z value <" << anZValue->GetName() << "> from composite position cylindrical creator <" << this->GetName() << ">" << eom;
        return;
    }

    void KSGenPositionFluxTube::AddMagneticField( KSMagneticField* aField )
    {
        fMagneticFields.push_back( aField );
    }

    void KSGenPositionFluxTube::CalculateField( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField )
    {
        aField = KThreeVector::sZero;
        KThreeVector tCurrentField = KThreeVector::sZero;
        for( size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++ )
        {
            fMagneticFields.at( tIndex )->CalculateField( aSamplePoint, aSampleTime, tCurrentField );
            aField += tCurrentField;
        }
        return;
    }

    void KSGenPositionFluxTube::InitializeComponent()
    {
        if( fPhiValue != NULL )
        {
            fPhiValue->Initialize();
        }
        if( fZValue != NULL )
        {
            fZValue->Initialize();
        }
        for ( auto tIndex: fMagneticFields )
        {
            tIndex->Initialize();
        }
        return;
    }
    void KSGenPositionFluxTube::DeinitializeComponent()
    {
        if( fPhiValue != NULL )
        {
            fPhiValue->Deinitialize();
        }
        if( fZValue != NULL )
        {
            fZValue->Deinitialize();
        }
        for ( auto tIndex: fMagneticFields )
        {
            tIndex->Deinitialize();
        }
        return;
    }

}
