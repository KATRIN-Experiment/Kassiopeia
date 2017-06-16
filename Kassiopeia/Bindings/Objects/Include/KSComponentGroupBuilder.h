#ifndef Kassiopeia_KSComponentGroupBuilder_h_
#define Kassiopeia_KSComponentGroupBuilder_h_

#include "KComplexElement.hh"
#include "KSComponentGroup.h"
#include "KToolbox.h"

namespace katrin
{

    typedef KComplexElement< Kassiopeia::KSComponentGroup > KSComponentGroupBuilder;

    template< >
    inline bool KSComponentGroupBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            std::string tName = aContainer->AsReference< std::string >();
            fObject->SetName( tName );
            return true;
        }
        return false;
    }

    template< >
    inline bool KSComponentGroupBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->Is< Kassiopeia::KSComponent >() == true )
        {
            aContainer->ReleaseTo( fObject, &Kassiopeia::KSComponentGroup::AddComponent);
            return true;
        }
        return false;
    }

}

#endif
