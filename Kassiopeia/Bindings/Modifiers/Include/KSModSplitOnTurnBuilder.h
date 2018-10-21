#ifndef Kassiopeia_KSModSplitOnTurnBuilder_h_
#define Kassiopeia_KSModSplitOnTurnBuilder_h_

#include "KComplexElement.hh"
#include "KSModSplitOnTurn.h"
#include "KToolbox.h"

using namespace Kassiopeia;

namespace katrin
{
    typedef KComplexElement< KSModSplitOnTurn > KSModSplitOnTurnBuilder;

    template< >
    inline bool KSModSplitOnTurnBuilder::AddAttribute(KContainer *aContainer)
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
                fObject->SetDirection( KSModSplitOnTurn::eForward );
                return true;
            }
            if( aContainer->AsReference< std::string >() == "backward" )
            {
                fObject->SetDirection( KSModSplitOnTurn::eBackward );
                return true;
            }
            if( aContainer->AsReference< std::string >() == "both" )
            {
                fObject->SetDirection( KSModSplitOnTurn::eForward | KSModSplitOnTurn::eBackward );
                return true;
            }
            return true;
        }
        return false;
    }
}

#endif // Kassiopeia_KSModSplitOnTurnBuilder_h_
