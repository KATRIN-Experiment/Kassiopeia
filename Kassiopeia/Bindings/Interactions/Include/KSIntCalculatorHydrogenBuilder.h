#ifndef Kassiopeia_KSIntCalculatorHydrogenBuilder_h_
#define Kassiopeia_KSIntCalculatorHydrogenBuilder_h_

#include "KField.h"
#include "KComplexElement.hh"
#include "KSIntCalculatorHydrogen.h"
#include "KSIntScatteringBuilder.h"


using namespace Kassiopeia;
namespace katrin
{

    class KSIntCalculatorHydrogenSet :
        public KSIntCalculatorSet
    {
        public:
            KSIntCalculatorHydrogenSet();
            virtual ~KSIntCalculatorHydrogenSet();

        public:
            void AddCalculator( KSIntCalculator* aCalculator );
            void ReleaseCalculators( KSIntScattering* aScattering );

        private:
            ;K_SET_GET( string, Name );
            ;K_SET_GET( bool, Elastic );
            ;K_SET_GET( bool, Excitation );
            ;K_SET_GET( bool, Ionisation );
            ;K_SET_GET( string, Molecule );
            vector< KSIntCalculator* > fCalculators;
    };

    typedef KComplexElement< KSIntCalculatorHydrogenSet > KSIntCalculatorHydrogenSetBuilder;

    template< >
    inline bool KSIntCalculatorHydrogenSetBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KSIntCalculatorHydrogenSet::SetName );
            return true;
        }
        if( aContainer->GetName() == "elastic" )
        {
            aContainer->CopyTo( fObject, &KSIntCalculatorHydrogenSet::SetElastic );
            return true;
        }
        if( aContainer->GetName() == "excitation" )
        {
            aContainer->CopyTo( fObject, &KSIntCalculatorHydrogenSet::SetExcitation );
            return true;
        }
        if( aContainer->GetName() == "ionisation" )
        {
            aContainer->CopyTo( fObject, &KSIntCalculatorHydrogenSet::SetIonisation );
            return true;
        }
        if( aContainer->GetName() == "molecule" )
        {
            aContainer->CopyTo( fObject, &KSIntCalculatorHydrogenSet::SetMolecule );
            return true;
        }
        return false;
    }

    template< >
    inline bool KSIntCalculatorHydrogenSetBuilder::End()
    {
        if( fObject->GetElastic() == true )
        {
            KSIntCalculator* aIntCalculator;

            aIntCalculator = new KSIntCalculatorHydrogenElastic();
            aIntCalculator->SetName( fObject->GetName() + "_elastic" );
            aIntCalculator->SetTag( fObject->GetName() );
            fObject->AddCalculator( aIntCalculator );

            aIntCalculator = new KSIntCalculatorHydrogenVib();
            aIntCalculator->SetName( fObject->GetName() + "_vib" );
            aIntCalculator->SetTag( fObject->GetName() );
            fObject->AddCalculator( aIntCalculator );

            aIntCalculator = new KSIntCalculatorHydrogenRot02();
            aIntCalculator->SetName( fObject->GetName() + "_rot_02" );
            aIntCalculator->SetTag( fObject->GetName() );
            fObject->AddCalculator( aIntCalculator );

            aIntCalculator = new KSIntCalculatorHydrogenRot13();
            aIntCalculator->SetName( fObject->GetName() + "_rot_13" );
            aIntCalculator->SetTag( fObject->GetName() );
            fObject->AddCalculator( aIntCalculator );

            if ( fObject->GetMolecule() == string ("tritium") )
            {
				aIntCalculator = new KSIntCalculatorHydrogenRot20();
				aIntCalculator->SetName( fObject->GetName() + "_rot_20" );
				aIntCalculator->SetTag( fObject->GetName() );
				fObject->AddCalculator( aIntCalculator );
            }

        }

        if( fObject->GetExcitation() == true )
        {
            KSIntCalculator* aIntCalculator;

            aIntCalculator = new KSIntCalculatorHydrogenExcitationB();
            aIntCalculator->SetName( fObject->GetName() + "_exc_b" );
            aIntCalculator->SetTag( fObject->GetName() );
            fObject->AddCalculator( aIntCalculator );

            aIntCalculator = new KSIntCalculatorHydrogenExcitationC();
            aIntCalculator->SetName( fObject->GetName() + "_exc_c" );
            aIntCalculator->SetTag( fObject->GetName() );
            fObject->AddCalculator( aIntCalculator );

            aIntCalculator = new KSIntCalculatorHydrogenDissoziation10();
            aIntCalculator->SetName( fObject->GetName() + "_diss_10" );
            aIntCalculator->SetTag( fObject->GetName() );
            fObject->AddCalculator( aIntCalculator );

            aIntCalculator = new KSIntCalculatorHydrogenDissoziation15();
            aIntCalculator->SetName( fObject->GetName() + "_diss_15" );
            aIntCalculator->SetTag( fObject->GetName() );
            fObject->AddCalculator( aIntCalculator );

            aIntCalculator = new KSIntCalculatorHydrogenExcitationElectronic();
            aIntCalculator->SetName( fObject->GetName() + "_exc_el" );
            aIntCalculator->SetTag( fObject->GetName() );
            fObject->AddCalculator( aIntCalculator );
        }

        if( fObject->GetIonisation() == true )
        {
            KSIntCalculator* aIntCalculator;

            aIntCalculator = new KSIntCalculatorHydrogenIonisation();
            aIntCalculator->SetName( fObject->GetName() + "_ionisation" );
            aIntCalculator->SetTag( fObject->GetName() );
            fObject->AddCalculator( aIntCalculator );
        }
        return true;
    }
}

#endif
