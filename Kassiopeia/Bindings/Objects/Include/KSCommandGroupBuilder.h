#ifndef Kassiopeia_KSCommandGroupBuilder_h_
#define Kassiopeia_KSCommandGroupBuilder_h_

#include "KComplexElement.hh"
#include "KSCommandGroup.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSCommandGroup > KSCommandGroupBuilder;

    template< >
    inline bool KSCommandGroupBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            std::string tName = aContainer->AsReference< std::string >();
            fObject->SetName( tName );
            return true;
        }
        if( aContainer->GetName() == "command" )
        {
            KSCommand* tCommand = KToolbox::GetInstance().Get< KSCommand >( aContainer->AsReference< std::string >() );
            fObject->AddCommand( tCommand->Clone() );
            return true;
        }
        return false;
    }

    template< >
    inline bool KSCommandGroupBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->Is< KSCommand >() == true )
        {
            /*KSCommand* tCommand;
            aContainer->ReleaseTo( tCommand );
            fObject->AddCommand( tCommand );*/
            aContainer->ReleaseTo( fObject, &KSCommandGroup::AddCommand );
            return true;
        }
        return false;
    }

}

#endif
