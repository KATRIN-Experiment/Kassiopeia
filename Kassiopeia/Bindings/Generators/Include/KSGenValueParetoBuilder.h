#ifndef Kassiopeia_KSGenValueParetoBuilder_h_
#define Kassiopeia_KSGenValueParetoBuilder_h_

#include "KComplexElement.hh"
#include "KSGenValuePareto.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSGenValuePareto > KSGenValueParetoBuilder;

    template< >
    inline bool KSGenValueParetoBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "value_min" )
        {
            aContainer->CopyTo( fObject, &KSGenValuePareto::SetValueMin );
            return true;
        }
        if( aContainer->GetName() == "value_max" )
        {
            aContainer->CopyTo( fObject, &KSGenValuePareto::SetValueMax );
            return true;
        }
        if( aContainer->GetName() == "slope" )
        {
            aContainer->CopyTo( fObject, &KSGenValuePareto::SetSlope );
            return true;
        }
        if( aContainer->GetName() == "cutoff" )
        {
            aContainer->CopyTo( fObject, &KSGenValuePareto::SetCutoff );
            return true;
        }
        if( aContainer->GetName() == "offset" )
        {
            aContainer->CopyTo( fObject, &KSGenValuePareto::SetOffset );
            return true;
        }
        return false;
    }

}

#endif
