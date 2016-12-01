#ifndef Kassiopeia_KSIntDecayBuilder_h_
#define Kassiopeia_KSIntDecayBuilder_h_

#include "KComplexElement.hh"
#include "KSIntDecay.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    class KSIntDecayCalculatorSet
    {
        public:
            KSIntDecayCalculatorSet();
            virtual ~KSIntDecayCalculatorSet();

        public:
            virtual void AddCalculator( KSIntDecayCalculator* aCalculator ) = 0;
            virtual void ReleaseCalculators( KSIntDecay* aDecay ) = 0;
    };

    typedef KComplexElement< KSIntDecay > KSIntDecayBuilder;

    template< >
    inline bool KSIntDecayBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "split" )
        {
            aContainer->CopyTo( fObject, &KSIntDecay::SetSplit );
            return true;
        }
        if( aContainer->GetName() == "calculator" )
        {
            KSIntDecayCalculator* tCalculator = KToolbox::GetInstance().Get< KSIntDecayCalculator >( aContainer->AsReference< std::string >() );
            fObject->AddCalculator( tCalculator );
            return true;
        }
        if( aContainer->GetName() == "calculators" )
        {
            std::vector< KSIntDecayCalculator* > aCalculatorVector = KToolbox::GetInstance().GetAll< KSIntDecayCalculator >( aContainer->AsReference< std::string >() );
            std::vector< KSIntDecayCalculator* >::iterator tIt;
            for( tIt = aCalculatorVector.begin(); tIt != aCalculatorVector.end(); tIt++ )
            {
                fObject->AddCalculator( (*tIt) );
            }
            return true;
        }
        if( aContainer->GetName() == "enhancement" )
        {
            aContainer->CopyTo( fObject, &KSIntDecay::SetEnhancement );
            return true;
        }
        return false;
    }

    template< >
    inline bool KSIntDecayBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->Is< KSIntDecayCalculator >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSIntDecay::AddCalculator );
            return true;
        }
        if( aContainer->Is< KSIntDecayCalculatorSet >() == true )
        {
            KSIntDecayCalculatorSet* tSet = NULL;
            aContainer->ReleaseTo( tSet );
            tSet->ReleaseCalculators( fObject );
            delete tSet;
            return true;
        }
        return false;
    }

}

#endif
