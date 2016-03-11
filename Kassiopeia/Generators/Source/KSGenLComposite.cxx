#include "KSGenLComposite.h"
#include "KSGeneratorsMessage.h"

namespace Kassiopeia
{

    KSGenLComposite::KSGenLComposite() :
            fLValue( NULL )
    {
    }
    KSGenLComposite::KSGenLComposite( const KSGenLComposite& aCopy ) :
            KSComponent(),
            fLValue( aCopy.fLValue )
    {
    }
    KSGenLComposite* KSGenLComposite::Clone() const
    {
        return new KSGenLComposite( *this );
    }
    KSGenLComposite::~KSGenLComposite()
    {
    }

    void KSGenLComposite::Dice( KSParticleQueue* aPrimaries )
    {
        KSParticle* tParticle;
        KSParticleQueue tParticles;
        KSParticleIt tParticleIt;

        double tLValue;
        vector< double > tLValues;
        vector< double >::iterator tLValueIt;

        fLValue->DiceValue( tLValues );

        for( tLValueIt = tLValues.begin(); tLValueIt != tLValues.end(); tLValueIt++ )
        {
            tLValue = static_cast<int>(*tLValueIt);
            for( tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++ )
            {
                tParticle = new KSParticle( **tParticleIt );
                tParticle->SetSecondQuantumNumber( tLValue );
                tParticles.push_back( tParticle );
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

    void KSGenLComposite::SetLValue( KSGenValue* anLValue )
    {
        if( fLValue == NULL )
        {
            fLValue = anLValue;
            return;
        }
        genmsg( eError ) << "cannot set L value <" << anLValue->GetName() << "> to composite L creator <" << this->GetName() << ">" << eom;
        return;
    }
    void KSGenLComposite::ClearLValue( KSGenValue* anLValue )
    {
        if( fLValue == anLValue )
        {
            fLValue = NULL;
            return;
        }
        genmsg( eError ) << "cannot clear L value <" << anLValue->GetName() << "> from composite L creator <" << this->GetName() << ">" << eom;
        return;
    }

    void KSGenLComposite::InitializeComponent()
    {
        if( fLValue != NULL )
        {
            fLValue->Initialize();
        }
        return;
    }
    void KSGenLComposite::DeinitializeComponent()
    {
        if( fLValue != NULL )
        {
            fLValue->Deinitialize();
        }
        return;
    }

    STATICINT sKSGenLCompositeDict =
        KSDictionary< KSGenLComposite >::AddCommand( &KSGenLComposite::SetLValue, &KSGenLComposite::ClearLValue, "set_L", "clear_L" );

}
