#include "KSGenEnergyComposite.h"
#include "KSGeneratorsMessage.h"

namespace Kassiopeia
{

    KSGenEnergyComposite::KSGenEnergyComposite() :
            fEnergyValue( NULL )
    {
    }
    KSGenEnergyComposite::KSGenEnergyComposite( const KSGenEnergyComposite& aCopy ) :
            KSComponent(),
            fEnergyValue( aCopy.fEnergyValue )
    {
    }
    KSGenEnergyComposite* KSGenEnergyComposite::Clone() const
    {
        return new KSGenEnergyComposite( *this );
    }
    KSGenEnergyComposite::~KSGenEnergyComposite()
    {
    }

    void KSGenEnergyComposite::Dice( KSParticleQueue* aPrimaries )
    {
        KSParticle* tParticle;
        KSParticleQueue tParticles;
        KSParticleIt tParticleIt;

        double tEnergyValue;
        vector< double > tEnergyValues;
        vector< double >::iterator tEnergyValueIt;

        fEnergyValue->DiceValue( tEnergyValues );

        for( tEnergyValueIt = tEnergyValues.begin(); tEnergyValueIt != tEnergyValues.end(); tEnergyValueIt++ )
        {
            tEnergyValue = *tEnergyValueIt;
            if( tEnergyValue < 0. )
            {
                genmsg( eWarning ) << "replacing negative energy value <"
                                   << tEnergyValue << "> in energy creator <"
                                   << this->GetName() << "> with zero" << eom;
                tEnergyValue = 0.;
            }
            if ( tEnergyValue == 0.0 )
            {
                genmsg( eWarning ) << "replacing zero energy value "
                                   << "in energy creator <"
                                   << this->GetName() << "> with 1e-10" << eom;
                tEnergyValue = 1.e-10;
            }
            for( tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++ )
            {
                tParticle = new KSParticle( **tParticleIt );
                tParticle->SetKineticEnergy_eV( tEnergyValue );
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

    void KSGenEnergyComposite::SetEnergyValue( KSGenValue* anEnergyValue )
    {
        if( fEnergyValue == NULL )
        {
            fEnergyValue = anEnergyValue;
            return;
        }
        genmsg( eError ) << "cannot set energy value <"
                         << anEnergyValue->GetName() << "> to composite energy creator <"
                         << this->GetName() << ">" << eom;
        return;
    }
    void KSGenEnergyComposite::ClearEnergyValue( KSGenValue* anEnergyValue )
    {
        if( fEnergyValue == anEnergyValue )
        {
            fEnergyValue = NULL;
            return;
        }
        genmsg( eError ) << "cannot clear energy value <"
                         << anEnergyValue->GetName() << "> from composite energy creator <"
                         << this->GetName() << ">" << eom;
        return;
    }

    void KSGenEnergyComposite::InitializeComponent()
    {
        if( fEnergyValue != NULL )
        {
            fEnergyValue->Initialize();
        }
        return;
    }
    void KSGenEnergyComposite::DeinitializeComponent()
    {
        if( fEnergyValue != NULL )
        {
            fEnergyValue->Deinitialize();
        }
        return;
    }

    STATICINT sKSGenEnergyCompositeDict =
        KSDictionary< KSGenEnergyComposite >::AddCommand( &KSGenEnergyComposite::SetEnergyValue,
                                                          &KSGenEnergyComposite::ClearEnergyValue,
                                                          "set_energy", "clear_energy" );
}
