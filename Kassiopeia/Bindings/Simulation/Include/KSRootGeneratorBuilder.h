#ifndef Kassiopeia_KSRootGeneratorBuilder_h_
#define Kassiopeia_KSRootGeneratorBuilder_h_

#include "KComplexElement.hh"
#include "KSRootGenerator.h"
#include "KSToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSRootGenerator > KSRootGeneratorBuilder;

    template< >
    inline bool KSRootGeneratorBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "set_generator" )
        {
            fObject->SetGenerator( KSToolbox::GetInstance()->GetObjectAs< KSGenerator >( aContainer->AsReference< string >() ) );
            return true;
        }
        return false;
    }

}

#endif
