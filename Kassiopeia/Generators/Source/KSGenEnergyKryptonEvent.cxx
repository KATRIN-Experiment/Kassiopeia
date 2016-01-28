#include "KSGeneratorsMessage.h"
#include "KSGenEnergyKryptonEvent.h"
#include "KSGenRelaxation.h"
#include "KSGenConversion.h"

#include "KSParticleFactory.h"

namespace Kassiopeia
{

    KSGenEnergyKryptonEvent::KSGenEnergyKryptonEvent() :
        fForceConversion( false ),
        fDoConversion( true ),
        fDoAuger( true ),
        fMyRelaxation( NULL ),
        fMyConversion( NULL )
    {
    }
    KSGenEnergyKryptonEvent::KSGenEnergyKryptonEvent( const KSGenEnergyKryptonEvent& aCopy ) :
        KSComponent(),
        fForceConversion( aCopy.fForceConversion ),
        fDoConversion( aCopy.fDoConversion ),
        fDoAuger( aCopy.fDoAuger ),
        fMyRelaxation( aCopy.fMyRelaxation ),
        fMyConversion( aCopy.fMyConversion )
    {
    }
    KSGenEnergyKryptonEvent* KSGenEnergyKryptonEvent::Clone() const
    {
        return new KSGenEnergyKryptonEvent( *this );
    }
    KSGenEnergyKryptonEvent::~KSGenEnergyKryptonEvent()
    {
    }

    void KSGenEnergyKryptonEvent::Dice( KSParticleQueue* aPrimaries )
    {

        KSParticle* tParticle;
        KSParticleQueue tParticles;
        KSParticleIt tParticleIt;


        for( tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++ )
        {


            //***********
            //conversions
            //***********

            vector< double > conversionElectronEnergy;
            vector< int > conversionVacancy;

            if( fDoConversion == true )
            {
                genmsg_debug( "creating a conversion electron" << fMyConversion << eom );

                fMyConversion->SetForceCreation( fForceConversion );
                while( conversionVacancy.size() == 0 )
                {
                    fMyConversion->CreateCE( conversionVacancy, conversionElectronEnergy );
                }

                for( unsigned int i = 0; i < conversionElectronEnergy.size(); i++ )
                {
                    genmsg_debug( "vacancy at: " << conversionVacancy.at(i) << ret );
                    genmsg_debug( "electron energy: " << conversionElectronEnergy.at(i) << eom );
                    fMyRelaxation->GetVacancies()->push_back( conversionVacancy.at( i ) );

                    tParticle = new KSParticle( **tParticleIt );
                    tParticle->SetLabel( "krypton_conversion" );
                    tParticle->SetKineticEnergy_eV( conversionElectronEnergy.at( i ) );
                    tParticles.push_back( tParticle );
                }
            }
            else
            {
                genmsg_debug( "conversion electron generation not activated" << eom );
            }

            //******
            //augers
            //******

            if( fDoAuger == true )
            {
                if( conversionElectronEnergy.size() != 0 )
                {
                    fMyRelaxation->ClearAugerEnergies();
                    fMyRelaxation->Relax();

                    for( unsigned int i = 0; i < fMyRelaxation->GetAugerEnergies().size(); i++ )
                    {
                        genmsg_debug( "auger energy: " << fMyRelaxation->GetAugerEnergies().at(i) << eom );
                        tParticle = new KSParticle( **tParticleIt );
                        tParticle->SetKineticEnergy_eV( fMyRelaxation->GetAugerEnergies().at( i ) );
                        tParticle->SetLabel( "krypton_auger" );
                        tParticles.push_back( tParticle );
                    }
                }
                else
                {
                    genmsg_debug( "no vacancy, therefore no auger electron can be produced" << eom );
                }
            }
            else
            {
                genmsg_debug( "auger electron production not activated" << eom );
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

    void KSGenEnergyKryptonEvent::SetForceConversion( bool aSetting )
    {
        fForceConversion = aSetting;
    }
    void KSGenEnergyKryptonEvent::SetDoConversion( bool aSetting )
    {
        fDoConversion = aSetting;
    }
    void KSGenEnergyKryptonEvent::SetDoAuger( bool aSetting )
    {
        fDoAuger = aSetting;
    }

    void KSGenEnergyKryptonEvent::InitializeComponent()
    {
        genmsg_debug( "reading conversion and relaxation data for krypton..." << eom );

        fMyConversion = new KSGenConversion();
        fMyConversion->Initialize( 83 );

        fMyRelaxation = new KSGenRelaxation();
        fMyRelaxation->Initialize( 83 );

        genmsg_debug( "...all data read, instances of conversion and relaxation class created" << eom );

        return;
    }
    void KSGenEnergyKryptonEvent::DeinitializeComponent()
    {
        delete fMyConversion;
        delete fMyRelaxation;
        return;
    }

}
