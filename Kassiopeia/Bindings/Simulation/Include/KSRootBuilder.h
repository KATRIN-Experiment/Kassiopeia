#ifndef Kassiopeia_KSRootBuilder_h_
#define Kassiopeia_KSRootBuilder_h_

#include "KComplexElement.hh"
#include "KSRoot.h"
#include "KSSimulation.h"
#include "KSToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSRoot > KSRootBuilder;

    template< >
    inline bool KSRootBuilder::Begin()
    {
        fObject = new KSRoot();
        return true;
    }

    template< >
    inline bool KSRootBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->Is< KSSimulation >() )
        {
            aContainer->ReleaseTo( fObject, &KSRoot::Execute );
            return true;
        }
        if( aContainer->Is< KSObject >() )
        {
            aContainer->ReleaseTo( KSToolbox::GetInstance(), &KSToolbox::AddObject );
            return true;
        }
        return false;
    }

    template< >
    inline bool KSRootBuilder::End()
    {
        delete fObject;
        return true;
    }

}

#endif
