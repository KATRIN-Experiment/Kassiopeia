#ifndef KGSURFACEBUILDER_HH_
#define KGSURFACEBUILDER_HH_

#include "KComplexElement.hh"

#include "KGCore.hh"
using namespace KGeoBag;

namespace katrin
{

    typedef KComplexElement< KGSurface > KGSurfaceBuilder;

    template< >
    inline bool KGSurfaceBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == "name" )
        {
            anAttribute->CopyTo( fObject, &KGSurface::SetName );
            return true;
        }
        if( anAttribute->GetName() == "node" )
        {
            KGSurface* tSource = KGInterface::GetInstance()->RetrieveSurface( anAttribute->AsReference< string >() );
            if( tSource == NULL )
            {
                return false;
            }

            KGSurface* tClone = tSource->CloneNode();
            tClone->SetName( fObject->GetName() );
            tClone->AddTags( fObject->GetTags() );

            fObject = tClone;
            Set( fObject );

            return true;
        }
        return false;
    }

    template< >
    inline bool KGSurfaceBuilder::AddElement( KContainer* anElement )
    {
        if( anElement->GetName() == "transformation" )
        {
            fObject->Transform( anElement->AsPointer< KTransformation >() );
            return true;
        }
        return false;
    }

}
#endif
