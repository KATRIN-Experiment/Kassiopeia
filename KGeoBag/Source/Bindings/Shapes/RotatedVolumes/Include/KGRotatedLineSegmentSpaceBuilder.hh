#ifndef KGROTATEDLINESEGMENTSPACEBUILDER_HH_
#define KGROTATEDLINESEGMENTSPACEBUILDER_HH_

#include "KGPlanarLineSegmentBuilder.hh"
#include "KGRotatedLineSegmentSpace.hh"

namespace katrin
{

//! Generates a volume enclosed by a surface of revolution produced from rotating a line segement.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_rotated_line_segment_space_model.png 400pt
//!
//! - Works with GeometryViewer: Yes.
//! - Works with MeshViewer: Yes.
//! @example
//! <rotated_line_segment_space name="rotated_line_segment_space" rotated_mesh_count="100" flattened_mesh_count="10" flattened_mesh_power="2.2">
//!     <line_segment x1="0.15" y1="0.1" x2="-0.1" y2="0.05" line_mesh_count="10" line_mesh_power="2.2"/>
//! </rotated_line_segment_space>
typedef KComplexElement<KGRotatedLineSegmentSpace> KGRotatedLineSegmentSpaceBuilder;

template<> inline bool KGRotatedLineSegmentSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGRotatedLineSegmentSpace::SetName);
        return true;
    }
    if (anAttribute->GetName() == "rotated_mesh_count") {
        anAttribute->CopyTo(fObject, &KGRotatedLineSegmentSpace::RotatedMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "flattened_mesh_count") {
        anAttribute->CopyTo(fObject, &KGRotatedLineSegmentSpace::FlattenedMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "flattened_mesh_power") {
        anAttribute->CopyTo(fObject, &KGRotatedLineSegmentSpace::FlattenedMeshPower);
        return true;
    }
    return false;
}

template<> inline bool KGRotatedLineSegmentSpaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "line_segment") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarLineSegment::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
