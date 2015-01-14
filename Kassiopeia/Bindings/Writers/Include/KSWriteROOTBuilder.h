#ifndef Kassiopeia_KSWriteROOTBuilder_h_
#define Kassiopeia_KSWriteROOTBuilder_h_

#include "KComplexElement.hh"
#include "KSWriteROOT.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSWriteROOT > KSWriteROOTBuilder;

    template< >
    inline bool KSWriteROOTBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "base" )
        {
            aContainer->CopyTo( fObject, &KSWriteROOT::SetBase );
            return true;
        }
        if( aContainer->GetName() == "path" )
        {
            aContainer->CopyTo( fObject, &KSWriteROOT::SetPath );
            return true;
        }
        return false;
    }

}
#endif
