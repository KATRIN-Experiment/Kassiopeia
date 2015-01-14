#include "KSGenDirectionSphericalComposite.h"
#include "KSGeneratorsMessage.h"

namespace Kassiopeia
{

    KSGenDirectionSphericalComposite::KSGenDirectionSphericalComposite() :
            fThetaValue( NULL ),
            fPhiValue( NULL ),
            fXAxis( KThreeVector::sXUnit ),
            fYAxis( KThreeVector::sYUnit ),
            fZAxis( KThreeVector::sZUnit )
    {
    }
    KSGenDirectionSphericalComposite::KSGenDirectionSphericalComposite( const KSGenDirectionSphericalComposite& aCopy ) :
            fThetaValue( aCopy.fThetaValue ),
            fPhiValue( aCopy.fPhiValue ),
            fXAxis( aCopy.fXAxis ),
            fYAxis( aCopy.fYAxis ),
            fZAxis( aCopy.fZAxis )
    {
    }
    KSGenDirectionSphericalComposite* KSGenDirectionSphericalComposite::Clone() const
    {
        return new KSGenDirectionSphericalComposite( *this );
    }
    KSGenDirectionSphericalComposite::~KSGenDirectionSphericalComposite()
    {
    }

    void KSGenDirectionSphericalComposite::Dice( KSParticleQueue* aPrimaries )
    {
        if ( !fThetaValue || !fPhiValue )
            genmsg( eError ) << "theta or phi value undefined in composite direction creator <" << this->GetName() << ">" << eom;

        KThreeVector tMomentum;

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
                    tMomentum = tParticle->GetMomentum().Magnitude() * (sin( tThetaValue ) * cos( tPhiValue ) * fXAxis + sin( tThetaValue ) * sin( tPhiValue ) * fYAxis + cos( tThetaValue ) * fZAxis);
                    tParticle->SetMomentum( tMomentum );
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

    void KSGenDirectionSphericalComposite::SetThetaValue( KSGenValue* anThetaValue )
    {
        if( fThetaValue == NULL )
        {
            fThetaValue = anThetaValue;
            return;
        }
        genmsg( eError ) << "cannot set theta value <" << anThetaValue->GetName() << "> to composite direction creator <" << this->GetName() << ">" << eom;
        return;
    }
    void KSGenDirectionSphericalComposite::ClearThetaValue( KSGenValue* anThetaValue )
    {
        if( fThetaValue == anThetaValue )
        {
            fThetaValue = NULL;
            return;
        }
        genmsg( eError ) << "cannot clear theta value <" << anThetaValue->GetName() << "> from composite direction creator <" << this->GetName() << ">" << eom;
        return;
    }

    void KSGenDirectionSphericalComposite::SetPhiValue( KSGenValue* aPhiValue )
    {
        if( fPhiValue == NULL )
        {
            fPhiValue = aPhiValue;
            return;
        }
        genmsg( eError ) << "cannot set phi value <" << aPhiValue->GetName() << "> to composite direction creator <" << this->GetName() << ">" << eom;
        return;
    }
    void KSGenDirectionSphericalComposite::ClearPhiValue( KSGenValue* anPhiValue )
    {
        if( fPhiValue == anPhiValue )
        {
            fPhiValue = NULL;
            return;
        }
        genmsg( eError ) << "cannot clear phi value <" << anPhiValue->GetName() << "> from composite direction creator <" << this->GetName() << ">" << eom;
        return;
    }

    void KSGenDirectionSphericalComposite::SetXAxis( const KThreeVector& anXAxis )
    {
        fXAxis = anXAxis;
        return;
    }
    void KSGenDirectionSphericalComposite::SetYAxis( const KThreeVector& anYAxis )
    {
        fYAxis = anYAxis;
        return;
    }
    void KSGenDirectionSphericalComposite::SetZAxis( const KThreeVector& anZAxis )
    {
        fZAxis = anZAxis;
        return;
    }

    void KSGenDirectionSphericalComposite::InitializeComponent()
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
    void KSGenDirectionSphericalComposite::DeinitializeComponent()
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

    static int sKSGenDirectionSphericalCompositeDict =
        KSDictionary< KSGenDirectionSphericalComposite >::AddCommand( &KSGenDirectionSphericalComposite::SetThetaValue, &KSGenDirectionSphericalComposite::ClearThetaValue, "set_theta", "clear_theta" ) +
        KSDictionary< KSGenDirectionSphericalComposite >::AddCommand( &KSGenDirectionSphericalComposite::SetPhiValue, &KSGenDirectionSphericalComposite::ClearPhiValue, "set_phi", "clear_phi" );

}
