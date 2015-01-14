#include "KSGenTimeComposite.h"
#include "KSGeneratorsMessage.h"

namespace Kassiopeia
{

    KSGenTimeComposite::KSGenTimeComposite() :
            fTimeValue( NULL )
    {
    }
    KSGenTimeComposite::KSGenTimeComposite( const KSGenTimeComposite& aCopy ) :
            fTimeValue( aCopy.fTimeValue )
    {
    }
    KSGenTimeComposite* KSGenTimeComposite::Clone() const
    {
        return new KSGenTimeComposite( *this );
    }
    KSGenTimeComposite::~KSGenTimeComposite()
    {
    }

    void KSGenTimeComposite::Dice( KSParticleQueue* aPrimaries )
    {
        KSParticle* tParticle;
        KSParticleQueue tParticles;
        KSParticleIt tParticleIt;

        double tTimeValue;
        vector< double > tTimeValues;
        vector< double >::iterator tTimeValueIt;

        fTimeValue->DiceValue( tTimeValues );

        for( tTimeValueIt = tTimeValues.begin(); tTimeValueIt != tTimeValues.end(); tTimeValueIt++ )
        {
            tTimeValue = *tTimeValueIt;
            for( tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++ )
            {
                tParticle = new KSParticle( **tParticleIt );
                tParticle->SetTime( tTimeValue );
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

    void KSGenTimeComposite::SetTimeValue( KSGenValue* anTimeValue )
    {
        if( fTimeValue == NULL )
        {
            fTimeValue = anTimeValue;
            return;
        }
        genmsg( eError ) << "cannot set time value <" << anTimeValue->GetName() << "> to composite time creator <" << this->GetName() << ">" << eom;
        return;
    }
    void KSGenTimeComposite::ClearTimeValue( KSGenValue* anTimeValue )
    {
        if( fTimeValue == anTimeValue )
        {
            fTimeValue = NULL;
            return;
        }
        genmsg( eError ) << "cannot clear time value <" << anTimeValue->GetName() << "> from composite time creator <" << this->GetName() << ">" << eom;
        return;
    }

    void KSGenTimeComposite::InitializeComponent()
    {
        if( fTimeValue != NULL )
        {
            fTimeValue->Initialize();
        }
        return;
    }
    void KSGenTimeComposite::DeinitializeComponent()
    {
        if( fTimeValue != NULL )
        {
            fTimeValue->Deinitialize();
        }
        return;
    }

    static int sKSGenTimeCompositeDict =
        KSDictionary< KSGenTimeComposite >::AddCommand( &KSGenTimeComposite::SetTimeValue, &KSGenTimeComposite::ClearTimeValue, "set_time", "clear_time" );

}
