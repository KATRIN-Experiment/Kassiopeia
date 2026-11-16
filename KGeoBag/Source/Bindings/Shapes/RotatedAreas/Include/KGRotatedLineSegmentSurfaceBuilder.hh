#ifndef KGROTATEDLINESEGMENTSURFACEBUILDER_HH_
#define KGROTATEDLINESEGMENTSURFACEBUILDER_HH_

#include "KGPlanarLineSegmentBuilder.hh"
#include "KGRotatedLineSegmentSurface.hh"

namespace katrin
{

//! This type generates the surface of revolution formed by a line that is rotated around the x-axis.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_rotated_line_segment_surface_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: Yes.
//! @example
//! <rotated_line_segment_surface name="rotated_line_segment_forward_surface" rotated_mesh_count="96">
//!     <line_segment x1="0.15" y1="0.1" x2="-0.1" y2="0.05" line_mesh_count="36" line_mesh_power="2.2"/>
//! </rotated_line_segment_surface>
typedef KComplexElement<KGRotatedLineSegmentSurface> KGRotatedLineSegmentSurfaceBuilder;

template<> inline bool KGRotatedLineSegmentSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGRotatedLineSegmentSurface::SetName);
        return true;
    }
    if (anAttribute->GetName() == "rotated_mesh_count") {
        anAttribute->CopyTo(fObject, &KGRotatedLineSegmentSurface::RotatedMeshCount);
        return true;
    }
    return false;
}

template<> inline bool KGRotatedLineSegmentSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "line_segment") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarLineSegment::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
