#ifndef Kassiopeia_KSGenValueSetBuilder_h_
#define Kassiopeia_KSGenValueSetBuilder_h_

#include "KComplexElement.hh"
#include "KSGenValueSet.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSGenValueSet > KSGenValueSetBuilder;

    template<>
    inline bool KSGenValueSetBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "value_start" )
        {
            aContainer->CopyTo( fObject, &KSGenValueSet::SetValueStart );
            return true;
        }
        if( aContainer->GetName() == "value_stop" )
        {
            aContainer->CopyTo( fObject, &KSGenValueSet::SetValueStop );
            return true;
        }
        if( aContainer->GetName() == "value_increment" )
        {
            aContainer->CopyTo( fObject, &KSGenValueSet::SetValueIncrement );
            return true;
        }
        if( aContainer->GetName() == "value_count" )
        {
            aContainer->CopyTo( fObject, &KSGenValueSet::SetValueCount );
            return true;
        }
        return false;
    }

}

#endif
