#ifndef Kassiopeia_KSIntCalculatorHydrogen_h_
#define Kassiopeia_KSIntCalculatorHydrogen_h_

#include "KSIntCalculator.h"

namespace Kassiopeia
{

    /////////////////////////////////////
    /////		Elastic	Base		/////
    /////////////////////////////////////

    class KSIntCalculatorHydrogenElasticBase :
        public KSComponentTemplate< KSIntCalculatorHydrogenElasticBase, KSIntCalculator >
    {
        public:
            KSIntCalculatorHydrogenElasticBase();
            virtual ~KSIntCalculatorHydrogenElasticBase();

        public:
            virtual void CalculateCrossSection( const KSParticle& anInitialParticle, double& aCrossSection );
            virtual void CalculateCrossSection( const double anEnergie, double& aCrossSection ) = 0;
            virtual void ExecuteInteraction( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aSecondaries );
            virtual void CalculateEloss( const double anEnergie, const double aTheta, double& anEloss ) = 0;

        protected:
            void CalculateTheta( const double anEnergy, double& aTheta );
            void CalculateDifferentialCrossSection( const double anEnergy, const double cosTheta, double& aCrossSection );

    };

    /////////////////////////////////
    /////		Elastic			/////
    /////////////////////////////////

    class KSIntCalculatorHydrogenElastic :
        public KSComponentTemplate< KSIntCalculatorHydrogenElastic, KSIntCalculatorHydrogenElasticBase >
    {
        public:
            KSIntCalculatorHydrogenElastic();
            KSIntCalculatorHydrogenElastic( const KSIntCalculatorHydrogenElastic& aCopy );
            KSIntCalculatorHydrogenElastic* Clone() const;
            virtual ~KSIntCalculatorHydrogenElastic();

        public:
            virtual void CalculateCrossSection( const double anEnergie, double& aCrossSection );
            virtual void CalculateEloss( const double anEnergie, const double aTheta, double& anEloss );

    };

    /////////////////////////////////
    /////		Vibration		/////
    /////////////////////////////////

    class KSIntCalculatorHydrogenVib :
        public KSComponentTemplate< KSIntCalculatorHydrogenVib, KSIntCalculatorHydrogenElasticBase >
    {
        public:
            KSIntCalculatorHydrogenVib();
            KSIntCalculatorHydrogenVib( const KSIntCalculatorHydrogenVib& aCopy );
            KSIntCalculatorHydrogenVib* Clone() const;
            virtual ~KSIntCalculatorHydrogenVib();

        public:
            virtual void CalculateCrossSection( const double anEnergie, double& aCrossSection );
            virtual void CalculateEloss( const double anEnergie, const double aTheta, double& anEloss );

    };

    /////////////////////////////////
    /////		Rot02			/////
    /////////////////////////////////

    class KSIntCalculatorHydrogenRot02 :
        public KSComponentTemplate< KSIntCalculatorHydrogenRot02, KSIntCalculatorHydrogenElasticBase >
    {
        public:
            KSIntCalculatorHydrogenRot02();
            KSIntCalculatorHydrogenRot02( const KSIntCalculatorHydrogenRot02& aCopy );
            KSIntCalculatorHydrogenRot02* Clone() const;
            virtual ~KSIntCalculatorHydrogenRot02();

        public:
            virtual void CalculateCrossSection( const double anEnergie, double& aCrossSection );
            virtual void CalculateEloss( const double anEnergie, const double aTheta, double& anEloss );

    };

    /////////////////////////////////
    /////		Rot13			/////
    /////////////////////////////////

    class KSIntCalculatorHydrogenRot13 :
        public KSComponentTemplate< KSIntCalculatorHydrogenRot13, KSIntCalculatorHydrogenElasticBase >
    {
        public:
            KSIntCalculatorHydrogenRot13();
            KSIntCalculatorHydrogenRot13( const KSIntCalculatorHydrogenRot13& aCopy );
            KSIntCalculatorHydrogenRot13* Clone() const;
            virtual ~KSIntCalculatorHydrogenRot13();

        public:
            virtual void CalculateCrossSection( const double anEnergie, double& aCrossSection );
            virtual void CalculateEloss( const double anEnergie, const double aTheta, double& anEloss );

    };

    /////////////////////////////////
    /////		Rot20			/////
    /////////////////////////////////

    class KSIntCalculatorHydrogenRot20 :
        public KSComponentTemplate< KSIntCalculatorHydrogenRot20, KSIntCalculatorHydrogenElasticBase >
    {
        public:
            KSIntCalculatorHydrogenRot20();
            KSIntCalculatorHydrogenRot20( const KSIntCalculatorHydrogenRot20& aCopy );
            KSIntCalculatorHydrogenRot20* Clone() const;
            virtual ~KSIntCalculatorHydrogenRot20();

        public:
            virtual void CalculateCrossSection( const double anEnergie, double& aCrossSection );
            virtual void CalculateEloss( const double anEnergie, const double aTheta, double& anEloss );

    };

    /////////////////////////////////////
    /////		Excitation Base		/////
    /////////////////////////////////////

    class KSIntCalculatorHydrogenExcitationBase :
        public KSComponentTemplate< KSIntCalculatorHydrogenExcitationBase, KSIntCalculator >
    {
        public:
            KSIntCalculatorHydrogenExcitationBase();
            virtual ~KSIntCalculatorHydrogenExcitationBase();

        public:
            virtual void CalculateCrossSection( const KSParticle& aParticle, double& aCrossSection );
            virtual void CalculateCrossSection( const double anEnergie, double& aCrossSection ) = 0;
            virtual void ExecuteInteraction( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aSecondaries );
            virtual void CalculateEloss( const double anEnergie, const double aTheta, double& anEloss ) = 0;

        protected:
            void CalculateTheta( const double anEnergy, double& aTheta );
            void CalculateDifferentialCrossSection( const double anEnergy, const double cosTheta, double& aCrossSection );

    };

    /////////////////////////////////
    /////		Excitation BC	/////
    /////////////////////////////////

    class KSIntCalculatorHydrogenExcitationBC :
        public KSComponentTemplate< KSIntCalculatorHydrogenExcitationBC, KSIntCalculatorHydrogenExcitationBase >
    {
        public:
            KSIntCalculatorHydrogenExcitationBC();
            KSIntCalculatorHydrogenExcitationBC( const KSIntCalculatorHydrogenExcitationBC& aCopy );
            KSIntCalculatorHydrogenExcitationBC* Clone() const;
            virtual ~KSIntCalculatorHydrogenExcitationBC();

        public:
            virtual void CalculateCrossSection( const double anEnergie, double& aCrossSection );
            virtual void CalculateEloss( const double anEnergie, const double aTheta, double& anEloss );

    };

    /////////////////////////////////
    /////		Ionisation		/////
    /////////////////////////////////

    class KSIntCalculatorHydrogenIonisationOld :
        public KSComponentTemplate< KSIntCalculatorHydrogenIonisationOld, KSIntCalculator >
    {
        public:
            KSIntCalculatorHydrogenIonisationOld();
            KSIntCalculatorHydrogenIonisationOld( const KSIntCalculatorHydrogenIonisationOld& aCopy );
            KSIntCalculatorHydrogenIonisationOld* Clone() const;
            virtual ~KSIntCalculatorHydrogenIonisationOld();

        public:
            void CalculateCrossSection( const KSParticle& aParticle, double& aCrossSection );
            void CalculateCrossSection( const double anEnergie, double& aCrossSection );
            void ExecuteInteraction( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aSecondaries );

        protected:
            virtual void InitializeComponent();

    };

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

#include "KField.h"

class KSIntCalculatorHydrogenData
{
        ;K_SET_GET(string, Name )
        ;
        ;K_SET_GET(bool, Elastic )
        ;
        ;K_SET_GET(bool, Excitation )
        ;
        ;K_SET_GET(bool, Ionisation )
        ;
};

#include "KComplexElement.hh"
#include "KToolbox.h"
#include "KSInteractionsMessage.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSIntCalculatorHydrogenData > KSIntCalculatorHydrogenBuilder;

    template< >
    inline bool KSIntCalculatorHydrogenBuilder::Begin()
    {

        fObject = new KSIntCalculatorHydrogenData;
        fObject->SetElastic( true );
        fObject->SetExcitation( true );
        fObject->SetIonisation( true );
        return true;
    }

    template< >
    inline bool KSIntCalculatorHydrogenBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KSIntCalculatorHydrogenData::SetName );
            return true;
        }
        if( aContainer->GetName() == "elastic" )
        {
            aContainer->CopyTo( fObject, &KSIntCalculatorHydrogenData::SetElastic );
            return true;
        }
        if( aContainer->GetName() == "excitation" )
        {
            aContainer->CopyTo( fObject, &KSIntCalculatorHydrogenData::SetExcitation );
            return true;
        }
        if( aContainer->GetName() == "ionisation" )
        {
            aContainer->CopyTo( fObject, &KSIntCalculatorHydrogenData::SetIonisation );
            return true;
        }
        return false;
    }

    template< >
    bool KSIntCalculatorHydrogenBuilder::End()
    {
        KToolbox& tToolBox = KToolbox::GetInstance();
        KSIntCalculator* aIntCalculator;

        if( fObject->GetElastic() == true )
        {
            aIntCalculator = new KSIntCalculatorHydrogenElastic();
            aIntCalculator->SetName( fObject->GetName() + "_elastic" );
            aIntCalculator->SetTag( fObject->GetName() );
            tToolBox.AddObject( aIntCalculator );

            aIntCalculator = new KSIntCalculatorHydrogenVib();
            aIntCalculator->SetName( fObject->GetName() + "_vib" );
            aIntCalculator->SetTag( fObject->GetName() );
            tToolBox.AddObject( aIntCalculator );

            aIntCalculator = new KSIntCalculatorHydrogenRot02();
            aIntCalculator->SetName( fObject->GetName() + "_rot02" );
            aIntCalculator->SetTag( fObject->GetName() );
            tToolBox.AddObject( aIntCalculator );

            aIntCalculator = new KSIntCalculatorHydrogenRot13();
            aIntCalculator->SetName( fObject->GetName() + "_rot13" );
            aIntCalculator->SetTag( fObject->GetName() );
            tToolBox.AddObject( aIntCalculator );

            aIntCalculator = new KSIntCalculatorHydrogenRot20();
            aIntCalculator->SetName( fObject->GetName() + "_rot20" );
            aIntCalculator->SetTag( fObject->GetName() );
            tToolBox.AddObject( aIntCalculator );
        }

        if( fObject->GetExcitation() == true )
        {
            //TOdo:
        }

        if( fObject->GetIonisation() == true )
        {
            aIntCalculator = new KSIntCalculatorHydrogenIonisationOld();
            aIntCalculator->SetName( fObject->GetName() + "_ionisation" );
            aIntCalculator->SetTag( fObject->GetName() );
            tToolBox.AddObject( aIntCalculator );
        }

        delete fObject;
        return true;
    }

}

#endif /* Kassiopeia_KSIntCalculatorHydrogen_h_ */
