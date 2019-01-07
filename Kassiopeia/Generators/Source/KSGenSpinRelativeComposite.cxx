#include "KSGenSpinRelativeComposite.h"
#include "KSGeneratorsMessage.h"

#include <math.h>

namespace Kassiopeia
{

    KSGenSpinRelativeComposite::KSGenSpinRelativeComposite() :
            fThetaValue( NULL ),
            fPhiValue( NULL )
    {
    }
    KSGenSpinRelativeComposite::KSGenSpinRelativeComposite( const KSGenSpinRelativeComposite& aCopy ) :
            KSComponent(),
            fThetaValue( aCopy.fThetaValue ),
            fPhiValue( aCopy.fPhiValue )
    {
    }
    KSGenSpinRelativeComposite* KSGenSpinRelativeComposite::Clone() const
    {
        return new KSGenSpinRelativeComposite( *this );
    }
    KSGenSpinRelativeComposite::~KSGenSpinRelativeComposite()
    {
    }

    void KSGenSpinRelativeComposite::Dice( KSParticleQueue* aPrimaries )
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
                    tParticle->SetAlignedSpin( cos( tThetaValue ) );
                    if ( std::isnan( tParticle->GetAlignedSpin() ) )
                    {
                        tParticle->SetAlignedSpin( 1. );
                    }
                    if (tParticle->GetAlignedSpin() < 0.99999 && tParticle->GetAlignedSpin() > -0.99999 )
                    {
                        tParticle->SetSpinAngle( tPhiValue );
                    }
                    else
                    {
                        tParticle->SetSpinAngle( 0 );
                    }
                    tParticle->RecalculateSpinGlobal();
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

    void KSGenSpinRelativeComposite::SetThetaValue( KSGenValue* anThetaValue )
    {
        if( fThetaValue == NULL )
        {
            fThetaValue = anThetaValue;
            return;
        }
        genmsg( eError ) << "cannot set theta value <" << anThetaValue->GetName() << "> to composite spin creator <" << this->GetName() << ">" << eom;
        return;
    }
    void KSGenSpinRelativeComposite::ClearThetaValue( KSGenValue* anThetaValue )
    {
        if( fThetaValue == anThetaValue )
        {
            fThetaValue = NULL;
            return;
        }
        genmsg( eError ) << "cannot clear theta value <" << anThetaValue->GetName() << "> from composite spin creator <" << this->GetName() << ">" << eom;
        return;
    }

    void KSGenSpinRelativeComposite::SetPhiValue( KSGenValue* aPhiValue )
    {
        if( fPhiValue == NULL )
        {
            fPhiValue = aPhiValue;
            return;
        }
        genmsg( eError ) << "cannot set phi value <" << aPhiValue->GetName() << "> to composite spin creator <" << this->GetName() << ">" << eom;
        return;
    }
    void KSGenSpinRelativeComposite::ClearPhiValue( KSGenValue* anPhiValue )
    {
        if( fPhiValue == anPhiValue )
        {
            fPhiValue = NULL;
            return;
        }
        genmsg( eError ) << "cannot clear phi value <" << anPhiValue->GetName() << "> from composite spin creator <" << this->GetName() << ">" << eom;
        return;
    }

    void KSGenSpinRelativeComposite::InitializeComponent()
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
    void KSGenSpinRelativeComposite::DeinitializeComponent()
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
        KSDictionary< KSGenSpinRelativeComposite >::AddCommand( &KSGenSpinRelativeComposite::SetThetaValue, &KSGenSpinRelativeComposite::ClearThetaValue, "set_theta", "clear_theta" ) +
        KSDictionary< KSGenSpinRelativeComposite >::AddCommand( &KSGenSpinRelativeComposite::SetPhiValue, &KSGenSpinRelativeComposite::ClearPhiValue, "set_phi", "clear_phi" );

}
