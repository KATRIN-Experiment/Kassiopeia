#ifndef KGEXTRUDEDARCSEGMENTSURFACEBUILDER_HH_
#define KGEXTRUDEDARCSEGMENTSURFACEBUILDER_HH_

#include "KGPlanarArcSegmentBuilder.hh"
#include "KGExtrudedArcSegmentSurface.hh"

namespace katrin
{

    typedef KComplexElement< KGExtrudedArcSegmentSurface > KGExtrudedArcSegmentSurfaceBuilder;

    template< >
    inline bool KGExtrudedArcSegmentSurfaceBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == "name" )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedArcSegmentSurface::SetName );
            return true;
        }
        if( anAttribute->GetName() == "zmin" )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedArcSegmentSurface::ZMin );
            return true;
        }
        if( anAttribute->GetName() == "zmax" )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedArcSegmentSurface::ZMax );
            return true;
        }
        if( anAttribute->GetName() == "extruded_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedArcSegmentSurface::ExtrudedMeshCount );
            return true;
        }
        if( anAttribute->GetName() == "extruded_mesh_power" )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedArcSegmentSurface::ExtrudedMeshPower );
            return true;
        }
        return false;
    }

    template< >
    inline bool KGExtrudedArcSegmentSurfaceBuilder::AddElement( KContainer* anElement )
    {
        if( anElement->GetName() == "arc_segment" )
        {
            anElement->CopyTo( fObject->Path().operator ->(), &KGPlanarArcSegment::CopyFrom );
            return true;
        }
        return false;
    }

}

#endif
