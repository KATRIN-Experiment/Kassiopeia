#include "KSGenSpinComposite.h"
#include "KSGeneratorsMessage.h"

#include <math.h>

namespace Kassiopeia
{

    KSGenSpinComposite::KSGenSpinComposite() :
            fThetaValue( NULL ),
            fPhiValue( NULL ),
            fXAxis( KThreeVector::sXUnit ),
            fYAxis( KThreeVector::sYUnit ),
            fZAxis( KThreeVector::sZUnit )
    {
    }
    KSGenSpinComposite::KSGenSpinComposite( const KSGenSpinComposite& aCopy ) :
            KSComponent(),
            fThetaValue( aCopy.fThetaValue ),
            fPhiValue( aCopy.fPhiValue ),
            fXAxis( aCopy.fXAxis ),
            fYAxis( aCopy.fYAxis ),
            fZAxis( aCopy.fZAxis )
    {
    }
    KSGenSpinComposite* KSGenSpinComposite::Clone() const
    {
        return new KSGenSpinComposite( *this );
    }
    KSGenSpinComposite::~KSGenSpinComposite()
    {
    }

    void KSGenSpinComposite::Dice( KSParticleQueue* aPrimaries )
    {
        if ( !fThetaValue || !fPhiValue )
            genmsg( eError ) << "theta or phi value undefined in composite direction creator <" << this->GetName() << ">" << eom;

        KThreeVector tSpin;

        KSParticle* tParticle;
        KSParticleIt tParticleIt;
        KSParticleQueue tParticles;

        double tThetaValue;
        vector< double > tThetaValues;
        vector< double >::iterator tThetaValueIt;

        double tPhiValue;
        vector< double > tPhiValues;
        vector< double >::iterator tPhiValueIt;

        fThetaValue->DiceValue( tThetaValues );
        fPhiValue->DiceValue( tPhiValues );

        for( tThetaValueIt = tThetaValues.begin(); tThetaValueIt != tThetaValues.end(); tThetaValueIt++ )
        {
            tThetaValue = (KConst::Pi() / 180.) * (*tThetaValueIt);
            for( tPhiValueIt = tPhiValues.begin(); tPhiValueIt != tPhiValues.end(); tPhiValueIt++ )
            {
                tPhiValue = (KConst::Pi() / 180.) * (*tPhiValueIt);
                for( tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++ )
                {
                    tParticle = new KSParticle( **tParticleIt );
                    tSpin = sin( tThetaValue ) * cos( tPhiValue ) * fXAxis + sin( tThetaValue ) * sin( tPhiValue ) * fYAxis + cos( tThetaValue ) * fZAxis;
                    tParticle->SetInitialSpin( tSpin );
                    KThreeVector LocalZ = tParticle->GetMagneticField() / tParticle->GetMagneticField().Magnitude();
                    KThreeVector LocalX  ( LocalZ.Z() - LocalZ.Y(), LocalZ.X() - LocalZ.Z(), LocalZ.Y() - LocalZ.X() );
                    LocalX = LocalX / LocalX.Magnitude();

                    //std::cout << "B: " << tParticle->GetMagneticField() << "\t\tZ(b): " << LocalZ << "\t\tX(b): " << LocalX << "\n";

                    tParticle->SetAlignedSpin( tSpin.Dot( LocalZ ) / tSpin.Magnitude() );
                    if ( std::isnan( tParticle->GetAlignedSpin() ) )
                    {
                        tParticle->SetAlignedSpin( 1. );
                        //std::cout << "*fixed NaN m (in GSC); B: " << tParticle->GetMagneticField() << "\n";
                    }
                    if (tParticle->GetAlignedSpin() < 0.99999 && tParticle->GetAlignedSpin() > -0.99999 )
                    {
                        tParticle->SetSpinAngle( acos( tSpin.Dot( LocalX ) / tSpin.Magnitude() / sqrt( 1 - tParticle->GetAlignedSpin() * tParticle->GetAlignedSpin() ) ) );
                    }
                    else
                    {
                        tParticle->SetSpinAngle( 0 );
                    }
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

    void KSGenSpinComposite::SetThetaValue( KSGenValue* anThetaValue )
    {
        if( fThetaValue == NULL )
        {
            fThetaValue = anThetaValue;
            return;
        }
        genmsg( eError ) << "cannot set theta value <" << anThetaValue->GetName() << "> to composite spin creator <" << this->GetName() << ">" << eom;
        return;
    }
    void KSGenSpinComposite::ClearThetaValue( KSGenValue* anThetaValue )
    {
        if( fThetaValue == anThetaValue )
        {
            fThetaValue = NULL;
            return;
        }
        genmsg( eError ) << "cannot clear theta value <" << anThetaValue->GetName() << "> from composite spin creator <" << this->GetName() << ">" << eom;
        return;
    }

    void KSGenSpinComposite::SetPhiValue( KSGenValue* aPhiValue )
    {
        if( fPhiValue == NULL )
        {
            fPhiValue = aPhiValue;
            return;
        }
        genmsg( eError ) << "cannot set phi value <" << aPhiValue->GetName() << "> to composite spin creator <" << this->GetName() << ">" << eom;
        return;
    }
    void KSGenSpinComposite::ClearPhiValue( KSGenValue* anPhiValue )
    {
        if( fPhiValue == anPhiValue )
        {
            fPhiValue = NULL;
            return;
        }
        genmsg( eError ) << "cannot clear phi value <" << anPhiValue->GetName() << "> from composite spin creator <" << this->GetName() << ">" << eom;
        return;
    }

    void KSGenSpinComposite::SetXAxis( const KThreeVector& anXAxis )
    {
        fXAxis = anXAxis;
        return;
    }
    void KSGenSpinComposite::SetYAxis( const KThreeVector& anYAxis )
    {
        fYAxis = anYAxis;
        return;
    }
    void KSGenSpinComposite::SetZAxis( const KThreeVector& anZAxis )
    {
        fZAxis = anZAxis;
        return;
    }

    void KSGenSpinComposite::InitializeComponent()
    {
        if( fThetaValue != NULL )
        {
            fThetaValue->Initialize();
        }
        if( fPhiValue != NULL )
        {
            fPhiValue->Initialize();
        }
        return;
    }
    void KSGenSpinComposite::DeinitializeComponent()
    {
        if( fThetaValue != NULL )
        {
            fThetaValue->Deinitialize();
        }
        if( fPhiValue != NULL )
        {
            fPhiValue->Deinitialize();
        }
        return;
    }

    STATICINT sKSGenDirectionSphericalCompositeDict =
        KSDictionary< KSGenSpinComposite >::AddCommand( &KSGenSpinComposite::SetThetaValue, &KSGenSpinComposite::ClearThetaValue, "set_theta", "clear_theta" ) +
        KSDictionary< KSGenSpinComposite >::AddCommand( &KSGenSpinComposite::SetPhiValue, &KSGenSpinComposite::ClearPhiValue, "set_phi", "clear_phi" );

}
