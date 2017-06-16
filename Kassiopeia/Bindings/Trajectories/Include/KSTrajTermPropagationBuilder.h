#ifndef Kassiopeia_KSTrajTermPropagationBuilder_h_
#define Kassiopeia_KSTrajTermPropagationBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajTermPropagation.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTrajTermPropagation > KSTrajTermPropagationBuilder;

    template< >
    inline bool KSTrajTermPropagationBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "direction" )
        {
            if( aContainer->AsReference< std::string >() == "forward" )
            {
                fObject->SetDirection( KSTrajTermPropagation::eForward );
                return true;
            }
            if( aContainer->AsReference< std::string >() == "backward" )
            {
                fObject->SetDirection( KSTrajTermPropagation::eBackward );
                return true;
            }
            return false;
        }
        return false;
    }

}
#endif
