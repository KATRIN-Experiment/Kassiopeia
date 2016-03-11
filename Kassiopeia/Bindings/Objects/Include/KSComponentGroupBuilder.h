#ifndef Kassiopeia_KSComponentGroupBuilder_h_
#define Kassiopeia_KSComponentGroupBuilder_h_

#include "KComplexElement.hh"
#include "KSComponentGroup.h"
#include "KSToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSComponentGroup > KSComponentGroupBuilder;

    template< >
    inline bool KSComponentGroupBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            string tName = aContainer->AsReference< string >();
            fObject->SetName( tName );
            return true;
        }
        return false;
    }

    template< >
    inline bool KSComponentGroupBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->Is< KSComponent >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSComponentGroup::AddComponent);
            return true;
        }
        return false;
    }

}

#endif
