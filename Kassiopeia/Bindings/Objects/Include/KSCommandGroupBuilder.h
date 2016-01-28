#ifndef Kassiopeia_KSCommandGroupBuilder_h_
#define Kassiopeia_KSCommandGroupBuilder_h_

#include "KComplexElement.hh"
#include "KSCommandGroup.h"
#include "KSToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSCommandGroup > KSCommandGroupBuilder;

    template< >
    inline bool KSCommandGroupBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            string tName = aContainer->AsReference< string >();
            fObject->SetName( tName );
            return true;
        }
        if( aContainer->GetName() == "command" )
        {
            KSCommand* tCommand = KSToolbox::GetInstance()->GetObjectAs< KSCommand >( aContainer->AsReference< string >() );
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
