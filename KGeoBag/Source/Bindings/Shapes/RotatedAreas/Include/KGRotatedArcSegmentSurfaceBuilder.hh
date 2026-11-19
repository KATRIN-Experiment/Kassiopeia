#ifndef KGROTATEDARCSEGMENTSURFACEBUILDER_HH_
#define KGROTATEDARCSEGMENTSURFACEBUILDER_HH_

#include "KGPlanarArcSegmentBuilder.hh"
#include "KGRotatedArcSegmentSurface.hh"

namespace katrin
{

//! This type generates the surface of revolution formed by an arc that is rotated around the x-axis.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_rotated_arc_segment_surface_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: Yes.
//! @example
//! <rotated_arc_segment_surface name="rotated_arc_segment_right_short_surface" rotated_mesh_count="128">
//!     <arc_segment x1="0.15" y1="0.1" x2="-0.1" y2="0.05" radius="0.3" right="true" short="true" arc_mesh_count="64"/>
//! </rotated_arc_segment_surface>
typedef KComplexElement<KGRotatedArcSegmentSurface> KGRotatedArcSegmentSurfaceBuilder;

template<> inline bool KGRotatedArcSegmentSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGRotatedArcSegmentSurface::SetName);
        return true;
    }
    if (anAttribute->GetName() == "rotated_mesh_count") {
        fObject->RotatedMeshCount(anAttribute->AsReference<unsigned int>());
        return true;
    }
    return false;
}

template<> inline bool KGRotatedArcSegmentSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "arc_segment") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarArcSegment::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
