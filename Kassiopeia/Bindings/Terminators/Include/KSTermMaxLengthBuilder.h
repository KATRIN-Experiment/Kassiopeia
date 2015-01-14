#ifndef Kassiopeia_KSTermMaxLengthBuilder_h_
#define Kassiopeia_KSTermMaxLengthBuilder_h_

#include "KComplexElement.hh"
#include "KSTermMaxLength.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTermMaxLength > KSTermMaxLengthBuilder;

    template< >
    inline bool KSTermMaxLengthBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "length" )
        {
            aContainer->CopyTo( fObject, &KSTermMaxLength::SetLength );
            return true;
        }
        return false;
    }

}

#endif
