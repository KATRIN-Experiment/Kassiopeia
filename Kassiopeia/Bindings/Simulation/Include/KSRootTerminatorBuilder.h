#ifndef Kassiopeia_KSRootTerminatorBuilder_h_
#define Kassiopeia_KSRootTerminatorBuilder_h_

#include "KComplexElement.hh"
#include "KSRootTerminator.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSRootTerminator > KSRootTerminatorBuilder;

    template< >
    inline bool KSRootTerminatorBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "add_terminator" )
        {
            fObject->AddTerminator( KToolbox::GetInstance().Get< KSTerminator >( aContainer->AsReference< std::string >() ) );
            return true;
        }
        return false;
    }

}

#endif
