#include "KESSScatteringModule.h"
#include "KESSScatteringCalculator.h"

#include "KRandom.h"
using katrin::KRandom;

#include "KSInteractionsMessage.h"

namespace Kassiopeia
{

    KESSScatteringModule::KESSScatteringModule() :
        fStep( NULL ),
        fKESSElasticCalculator(),
        fKESSInelasticCalculator(),
        fDeadLayer( 0.0 ),
        fMeanFreePathInvert( 0. ),
        fDepositedEnergy( 0.0 ),
        fDepositedEnergyDeadLayer( 0.0 ),
        fScatteringPolarAngleToZ( 0.0 ),
        fScatteringAzimuthalAngleToX( 0.0 )
    {
    }

    KESSScatteringModule::~KESSScatteringModule()
    {
    }

    void KESSScatteringModule::SetStep( KSStep* aStep )
    {
        fStep = aStep;
        fKESSElasticCalculator->SetStep( aStep );
        fKESSInelasticCalculator->SetStep( aStep );
        return;
    }

    void KESSScatteringModule::SetElasticScatteringCalculator( KESSScatteringCalculator* aScatteringCalculator )
    {
        fKESSElasticCalculator = aScatteringCalculator;
        fKESSElasticCalculator->SetStep( fStep );
        fKESSElasticCalculator->SetName( this->GetName() + "_elastic" );
        fKESSElasticCalculator->SetModule( this );
    }

    void KESSScatteringModule::SetInelasticScatteringCalculator( KESSScatteringCalculator* aScatteringCalculator )
    {
        fKESSInelasticCalculator = aScatteringCalculator;
        fKESSInelasticCalculator->SetStep( fStep );
        fKESSInelasticCalculator->SetName( this->GetName() + "_inelastic" );
        fKESSInelasticCalculator->SetModule( this );
    }

    void KESSScatteringModule::ExecuteSpaceInteraction()
    {
        KESSScatteringCalculator* tScatteringCalculator = NULL;
        if( KRandom::GetInstance().Uniform() <= (1. / fKESSElasticCalculator->GetMeanFreePath()) / fMeanFreePathInvert )
        {
            tScatteringCalculator = fKESSElasticCalculator;
        }
        else
        {
            tScatteringCalculator = fKESSInelasticCalculator;
        }
        intmsg_debug( "kess scattering with module <" << tScatteringCalculator->GetName() << ">" << eom );
        fStep->SetSpaceInteractionName( tScatteringCalculator->GetName() );
        tScatteringCalculator->Execute();
    }

    void KESSScatteringModule::Reset()
    {
        fMeanFreePath = 0.;
        fMeanFreePathPtr = &KSSpaceInteraction::RecalculateMeanFreePath;
        fMeanFreePathInvert = 0.;
        fKESSElasticCalculator->Reset();
        fKESSInelasticCalculator->Reset();
        fDepositedEnergy = 0.0;
        fDepositedEnergyDeadLayer = 0.0;
        return;
    }

    void KESSScatteringModule::RecalculateMeanFreePath()
    {
        fMeanFreePathInvert = (1. / fKESSElasticCalculator->GetMeanFreePath()) + (1. / fKESSInelasticCalculator->GetMeanFreePath());
        fMeanFreePath = (1. / fMeanFreePathInvert);

        fMeanFreePathPtr = &KSSpaceInteraction::DoNothing;
    }

    bool KESSScatteringModule::CheckComposition()
    {
        if( fKESSElasticCalculator == 0 )
        {
            intmsg( eWarning ) << "KESSScatteringModule::CheckComposition" << ret << "no scattering calculator assigned for elastic scattering module <" << GetName() << ">" << eom;
            return false;
        }
        else if( fKESSInelasticCalculator == 0 )
        {
            intmsg( eWarning ) << "KESSScatteringModule::CheckComposition" << ret << "no scattering calculator assigned for inelastic module <" << GetName() << ">" << eom;
            return false;
        }
        return true;
    }

} /* namespace Kassiopeia */

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////                                                   /////
/////  BBBB   U   U  IIIII  L      DDDD   EEEEE  RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB   U   U    I    L      D   D  EE     RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB    UUU   IIIII  LLLLL  DDDD   EEEEE  R   R  /////
/////                                                   /////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

#include "KSToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KESSScatteringModuleBuilder::~KComplexElement()
    {
    }

    static int sKESSScatteringModuleStructure = KESSScatteringModuleBuilder::Attribute< string >( "name" ) + KESSScatteringModuleBuilder::Attribute< double >( "deadlayer_in_global_coordinates" );

    static int sToolboxKESSScatteringModule = KSToolboxBuilder::ComplexElement< KESSScatteringModule >( "kess_scattering_module" );


    static const int sKSKESSScatteringModuleDict =
        KSDictionary< KESSScatteringModule >::AddOutput( &KESSScatteringModule::GetDepositedEnergy, "deposited_energy" ) +
        KSDictionary< KESSScatteringModule >::AddOutput( &KESSScatteringModule::GetDepositedEnergyDeadLayer, "deposited_energy_deadlayer" ) +
        KSDictionary< KESSScatteringModule >::AddOutput( &KESSScatteringModule::GetScatteringPolarAngleToZ, "scattering_polar" ) +
        KSDictionary< KESSScatteringModule >::AddOutput( &KESSScatteringModule::GetScatteringAzimuthalAngleToX, "scattering_polar" );


}
