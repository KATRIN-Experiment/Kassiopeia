#ifndef KGROTATEDARCSEGMENTSPACEBUILDER_HH_
#define KGROTATEDARCSEGMENTSPACEBUILDER_HH_

#include "KGPlanarArcSegmentBuilder.hh"
#include "KGRotatedArcSegmentSpace.hh"

namespace katrin
{

//! Generates a volume enclosed by a surface of revolution produced from rotating an arc segment.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_rotated_arc_segment_space_model.png 400pt
//!
//! - Works with GeometryViewer: Yes.
//! - Works with MeshViewer: Yes.
//! @example
//! <rotated_arc_segment_space name="rotated_arc_segment_space" rotated_mesh_count="128" flattened_mesh_count="10" flattened_mesh_power="1.5">
//!     <arc_segment x1="0.15" y1="0.1" x2="-0.1" y2="0.03" radius="0.5" right="true" short="true" arc_mesh_count="64"/>
//! </rotated_arc_segment_space>
typedef KComplexElement<KGRotatedArcSegmentSpace> KGRotatedArcSegmentSpaceBuilder;

template<> inline bool KGRotatedArcSegmentSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGRotatedArcSegmentSpace::SetName);
        return true;
    }
    if (anAttribute->GetName() == "rotated_mesh_count") {
        anAttribute->CopyTo(fObject, &KGRotatedArcSegmentSpace::RotatedMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "flattened_mesh_count") {
        anAttribute->CopyTo(fObject, &KGRotatedArcSegmentSpace::FlattenedMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "flattened_mesh_power") {
        anAttribute->CopyTo(fObject, &KGRotatedArcSegmentSpace::FlattenedMeshPower);
        return true;
    }
    return false;
}

template<> inline bool KGRotatedArcSegmentSpaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "arc_segment") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarArcSegment::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
