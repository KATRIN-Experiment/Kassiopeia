#ifndef Kassiopeia_KSGenValueFormulaBuilder_h_
#define Kassiopeia_KSGenValueFormulaBuilder_h_

#include "KComplexElement.hh"
#include "KSGenValueFormula.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSGenValueFormula > KSGenValueFormulaBuilder;

    template< >
    inline bool KSGenValueFormulaBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "value_min" )
        {
            aContainer->CopyTo( fObject, &KSGenValueFormula::SetValueMin );
            return true;
        }
        if( aContainer->GetName() == "value_max" )
        {
            aContainer->CopyTo( fObject, &KSGenValueFormula::SetValueMax );
            return true;
        }
        if( aContainer->GetName() == "value_formula" )
        {
            aContainer->CopyTo( fObject, &KSGenValueFormula::SetValueFormula );
            return true;
        }
        return false;
    }

}

#endif
