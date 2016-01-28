#ifndef Kassiopeia_KSIntDecayCalculatorGlukhovSpontaneousBuilder_h_
#define Kassiopeia_KSIntDecayCalculatorGlukhovSpontaneousBuilder_h_

#include "KComplexElement.hh"
#include "KSIntDecayCalculatorGlukhovSpontaneous.h"
#include "KSToolbox.h"


using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSIntDecayCalculatorGlukhovSpontaneous > KSIntDecayCalculatorGlukhovSpontaneousBuilder;

    template< >
    inline bool KSIntDecayCalculatorGlukhovSpontaneousBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "target_pid" )
        {
            aContainer->CopyTo( fObject, &KSIntDecayCalculatorGlukhovSpontaneous::SetTargetPID );
            return true;
        }
        if( aContainer->GetName() == "min_pid" )
        {
            aContainer->CopyTo( fObject, &KSIntDecayCalculatorGlukhovSpontaneous::SetminPID );
            return true;
        }
        if( aContainer->GetName() == "max_pid" )
        {
            aContainer->CopyTo( fObject, &KSIntDecayCalculatorGlukhovSpontaneous::SetminPID );
            return true;
        }

        return false;
    }

}

#endif
