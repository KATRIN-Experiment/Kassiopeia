#ifndef KGROTATEDLINESEGMENTSPACEBUILDER_HH_
#define KGROTATEDLINESEGMENTSPACEBUILDER_HH_

#include "KGPlanarLineSegmentBuilder.hh"
#include "KGRotatedLineSegmentSpace.hh"

namespace katrin
{

    typedef KComplexElement< KGRotatedLineSegmentSpace > KGRotatedLineSegmentSpaceBuilder;

    template< >
    inline bool KGRotatedLineSegmentSpaceBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == "name" )
        {
            anAttribute->CopyTo( fObject, &KGRotatedLineSegmentSpace::SetName );
            return true;
        }
        if( anAttribute->GetName() == "rotated_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGRotatedLineSegmentSpace::RotatedMeshCount );
            return true;
        }
        if( anAttribute->GetName() == "flattened_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGRotatedLineSegmentSpace::FlattenedMeshCount );
            return true;
        }
        if( anAttribute->GetName() == "flattened_mesh_power" )
        {
            anAttribute->CopyTo( fObject, &KGRotatedLineSegmentSpace::FlattenedMeshPower );
            return true;
        }
        return false;
    }

    template< >
    inline bool KGRotatedLineSegmentSpaceBuilder::AddElement( KContainer* anElement )
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
