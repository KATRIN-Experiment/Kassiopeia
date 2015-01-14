#ifndef Kassiopeia_KESSScatteringModule_h_
#define Kassiopeia_KESSScatteringModule_h_

#include "KSSpaceInteraction.h"

namespace Kassiopeia
{
    class KESSScatteringCalculator;

    class KESSScatteringModule :
        public KSComponentTemplate< KESSScatteringModule, KSSpaceInteraction >
    {
        public:
            KESSScatteringModule();
            virtual ~KESSScatteringModule();

            //***********
            //composition
            //***********

        public:
            virtual void SetStep( KSStep* aStep );
            void SetElasticScatteringCalculator( KESSScatteringCalculator* const aScatteringCalculator );
            void SetInelasticScatteringCalculator( KESSScatteringCalculator* const aScatteringCalculator );

            void SetDeadLayer( double DeadLayer );
            double GetDeadLayer() const;

            bool CheckComposition();

        private:
            KSStep* fStep;
            KESSScatteringCalculator* fKESSElasticCalculator;
            KESSScatteringCalculator* fKESSInelasticCalculator;

            double fDeadLayer; //in nm

            //***********
            //  action
            //***********

        public:
            virtual void ExecuteSpaceInteraction();
            virtual void Reset();
            virtual void RecalculateMeanFreePath();

        private:
            double fMeanFreePathInvert;


            //******************
            //  outputvariables
            //******************

        public:
            void SetDepositedEnergy( const double& anEnergy );
            const double& GetDepositedEnergy() const;

            void SetDepositedEnergyDeadLayer( const double& anEnergy );
            const double& GetDepositedEnergyDeadLayer() const;

            void SetScatteringPolarAngleToZ( const double& anAngle );
            const double& GetScatteringPolarAngleToZ() const;

            void SetScatteringAzimuthalAngleToX( const double& anAngle );
            const double& GetScatteringAzimuthalAngleToX() const;


        private:
            double fDepositedEnergy;
            double fDepositedEnergyDeadLayer;
            double fScatteringPolarAngleToZ;
            double fScatteringAzimuthalAngleToX;

    };

    inline void KESSScatteringModule::SetDeadLayer( double DeadLayer )
    {
        fDeadLayer = DeadLayer;
    }

    inline double KESSScatteringModule::GetDeadLayer() const
    {
        return fDeadLayer;
    }

    inline void KESSScatteringModule::SetDepositedEnergy( const double& anEnergy )
    {
        fDepositedEnergy = anEnergy;
    }
    inline const double& KESSScatteringModule::GetDepositedEnergy() const
    {
        return fDepositedEnergy;
    }
    inline void KESSScatteringModule::SetDepositedEnergyDeadLayer( const double& anEnergy )
    {
        fDepositedEnergyDeadLayer = anEnergy;
    }
    inline const double& KESSScatteringModule::GetDepositedEnergyDeadLayer() const
    {
        return fDepositedEnergyDeadLayer;
    }
    inline void KESSScatteringModule::SetScatteringPolarAngleToZ( const double& anAngle )
    {
        fScatteringPolarAngleToZ = anAngle;
    }
    inline const double& KESSScatteringModule::GetScatteringPolarAngleToZ() const
    {
        return fScatteringPolarAngleToZ;
    }
    inline void KESSScatteringModule::SetScatteringAzimuthalAngleToX( const double& anAngle )
    {
        fScatteringAzimuthalAngleToX = anAngle;
    }
    inline const double& KESSScatteringModule::GetScatteringAzimuthalAngleToX() const
    {
        return fScatteringAzimuthalAngleToX;
    }

}

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

#include "KComplexElement.hh"
#include "KESSInelasticBetheFano.h"
#include "KESSInelasticPenn.h"
#include "KESSElasticElsepa.h"
#include "KESSPhotoAbsorbtion.h"
#include "KESSRelaxation.h"
#include "KSInteractionsMessage.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KESSScatteringModule > KESSScatteringModuleBuilder;

    template< >
    inline bool KESSScatteringModuleBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }

        if( aContainer->GetName() == "deadlayer_in_global_coordinates" )
        {
            aContainer->CopyTo( fObject, &KESSScatteringModule::SetDeadLayer );
            intmsg_debug( "KESSScatteringModuleBuilder::SetAttribute: The dead layer is set." << eom );

            return true;
        }

        return false;
    }

    template< >
    inline bool KESSScatteringModuleBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->Is< KESSElasticElsepa >() )
        {
            aContainer->ReleaseTo( fObject, &KESSScatteringModule::SetElasticScatteringCalculator );
            return true;
        }
        if( aContainer->Is< KESSInelasticBetheFano >() )
        {
            aContainer->ReleaseTo( fObject, &KESSScatteringModule::SetInelasticScatteringCalculator );
            return true;
        }
        if( aContainer->Is< KESSInelasticPenn >() )
        {
            aContainer->ReleaseTo( fObject, &KESSScatteringModule::SetInelasticScatteringCalculator );
            return true;
        }
        return false;
    }

    template< >
    inline bool KESSScatteringModuleBuilder::End()
    {
        if( fObject->CheckComposition() == false )
        {
            intmsg( eWarning ) << "KESSScatteringModuleBuilder::End: your scattering module is not complete." << eom;
            return false;
        }
        return true;
    }

}

#endif /* KESSSCATTERINGMODULE_H_ */
