#ifndef KGROTATEDPOLYLINESURFACEBUILDER_HH_
#define KGROTATEDPOLYLINESURFACEBUILDER_HH_

#include "KGPlanarPolyLineBuilder.hh"
#include "KGRotatedPolyLineSurface.hh"

namespace katrin
{

    typedef KComplexElement< KGRotatedPolyLineSurface > KGRotatedPolyLineSurfaceBuilder;

    template< >
    inline bool KGRotatedPolyLineSurfaceBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == "name" )
        {
            anAttribute->CopyTo( fObject, &KGRotatedPolyLineSurface::SetName );
            return true;
        }
        if( anAttribute->GetName() == "rotated_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGRotatedPolyLineSurface::RotatedMeshCount );
            return true;
        }
        return false;
    }

    template< >
    inline bool KGRotatedPolyLineSurfaceBuilder::AddElement( KContainer* anElement )
    {
        if( anElement->GetName() == "poly_line" )
        {
            anElement->CopyTo( fObject->Path().operator ->(), &KGPlanarPolyLine::CopyFrom );
            return true;
        }
        return false;
    }

}

#endif
