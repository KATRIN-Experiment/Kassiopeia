#ifndef KGROTATEDLINESEGMENTSURFACEBUILDER_HH_
#define KGROTATEDLINESEGMENTSURFACEBUILDER_HH_

#include "KGPlanarLineSegmentBuilder.hh"
#include "KGRotatedLineSegmentSurface.hh"

namespace katrin
{

    typedef KComplexElement< KGRotatedLineSegmentSurface > KGRotatedLineSegmentSurfaceBuilder;

    template< >
    inline bool KGRotatedLineSegmentSurfaceBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == "name" )
        {
            anAttribute->CopyTo( fObject, &KGRotatedLineSegmentSurface::SetName );
            return true;
        }
        if( anAttribute->GetName() == "rotated_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGRotatedLineSegmentSurface::RotatedMeshCount );
            return true;
        }
        return false;
    }

    template< >
    inline bool KGRotatedLineSegmentSurfaceBuilder::AddElement( KContainer* anElement )
    {
        if( anElement->GetName() == "line_segment" )
        {
            anElement->CopyTo( fObject->Path().operator ->(), &KGPlanarLineSegment::CopyFrom );
            return true;
        }
        return false;
    }

}

#endif
