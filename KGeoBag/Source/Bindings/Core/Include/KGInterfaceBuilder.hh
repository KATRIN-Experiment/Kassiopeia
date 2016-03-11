#ifndef KGINTERFACEBUILDER_HH_
#define KGINTERFACEBUILDER_HH_

#include "KGCore.hh"

#include "KComplexElement.hh"
using namespace KGeoBag;

namespace katrin
{

    typedef KComplexElement< KGInterface > KGInterfaceBuilder;

    template< >
    inline bool KGInterfaceBuilder::Begin()
    {
        fObject = KGeoBag::KGInterface::GetInstance();
        return true;
    }

    template< >
    inline bool KGInterfaceBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "reset" )
        {
            if( aContainer->AsReference< bool >() == true )
            {
                fObject = KGeoBag::KGInterface::DeleteInstance();
                fObject = KGeoBag::KGInterface::GetInstance();
            }
            return true;
        }
        return false;
    }

    template< >
    inline bool KGInterfaceBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->Is< KGSurface >() )
        {
            aContainer->ReleaseTo( fObject, &KGInterface::InstallSurface );
            return true;
        }
        if( aContainer->Is< KGArea >() )
        {
            KGArea* tArea = NULL;
            aContainer->ReleaseTo( tArea );
            KGSurface* tSurface = new KGSurface();
            tSurface->SetName( tArea->GetName() );
            tSurface->SetTags( tArea->GetTags() );
            tSurface->Area( tArea );
            fObject->InstallSurface( tSurface );
            return true;
        }
        if( aContainer->Is< KGSpace >() )
        {
            aContainer->ReleaseTo( fObject, &KGInterface::InstallSpace );
            return true;
        }
        if( aContainer->Is< KGVolume >() )
        {
            KGVolume* tVolume = NULL;
            aContainer->ReleaseTo( tVolume );
            KGSpace* tSpace = new KGSpace();
            tSpace->SetName( tVolume->GetName() );
            tSpace->SetTags( tVolume->GetTags() );
            tSpace->Volume( tVolume );
            fObject->InstallSpace( tSpace );
            return true;
        }
        return true;
    }

    template< >
    inline bool KGInterfaceBuilder::End()
    {
        fObject = NULL;
        return true;
    }

}

#endif
