#ifndef Kassiopeia_KSTrajTermSynchrotronBuilder_h_
#define Kassiopeia_KSTrajTermSynchrotronBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajTermSynchrotron.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTrajTermSynchrotron > KSTrajTermSynchrotronBuilder;

    template< >
    inline bool KSTrajTermSynchrotronBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "enhancement" )
        {
            aContainer->CopyTo( fObject, &KSTrajTermSynchrotron::SetEnhancement );
            return true;
        }
        if( aContainer->GetName() == "old_methode" )
        {
            aContainer->CopyTo( fObject, &KSTrajTermSynchrotron::SetOldMethode );
            return true;
        }
        return false;
    }

}
#endif
