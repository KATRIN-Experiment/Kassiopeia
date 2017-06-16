#ifndef KGROTATEDARCSEGMENTSPACEBUILDER_HH_
#define KGROTATEDARCSEGMENTSPACEBUILDER_HH_

#include "KGPlanarArcSegmentBuilder.hh"
#include "KGRotatedArcSegmentSpace.hh"

namespace katrin
{

    typedef KComplexElement< KGRotatedArcSegmentSpace > KGRotatedArcSegmentSpaceBuilder;

    template< >
    inline bool KGRotatedArcSegmentSpaceBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == "name" )
        {
            anAttribute->CopyTo( fObject, &KGRotatedArcSegmentSpace::SetName );
            return true;
        }
        if( anAttribute->GetName() == "rotated_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGRotatedArcSegmentSpace::RotatedMeshCount );
            return true;
        }
        if( anAttribute->GetName() == "flattened_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGRotatedArcSegmentSpace::FlattenedMeshCount );
            return true;
        }
        if( anAttribute->GetName() == "flattened_mesh_power" )
        {
            anAttribute->CopyTo( fObject, &KGRotatedArcSegmentSpace::FlattenedMeshPower );
            return true;
        }
        return false;
    }

    template< >
    inline bool KGRotatedArcSegmentSpaceBuilder::AddElement( KContainer* anElement )
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
