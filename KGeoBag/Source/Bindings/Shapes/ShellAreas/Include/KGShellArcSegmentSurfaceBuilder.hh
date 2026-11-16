#ifndef KGSHELLARCSEGMENTSURFACEBUILDER_HH_
#define KGSHELLARCSEGMENTSURFACEBUILDER_HH_

#include "KGPlanarArcSegmentBuilder.hh"
#include "KGShellArcSegmentSurface.hh"

namespace katrin
{

//! This produces an angularly limited portion of a surface of revolution from an arc that is rotated around the local
//! x-axis.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_shell_arc_segment_surface_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: Yes.
//! @example
//! <shell_arc_segment_surface name="shell_arc_segment_right_short_surface" shell_mesh_count="128" shell_mesh_power="6">
//!     <arc_segment x1="0.15" y1="0.1" x2="-0.1" y2="0.05" radius="0.3" right="true" short="true" arc_mesh_count="64"/>
//! </shell_arc_segment_surface>
typedef KComplexElement<KGShellArcSegmentSurface> KGShellArcSegmentSurfaceBuilder;

template<> inline bool KGShellArcSegmentSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGShellArcSegmentSurface::SetName);
        return true;
    }
    if (anAttribute->GetName() == "angle_start") {
        anAttribute->CopyTo(fObject, &KGShellArcSegmentSurface::AngleStart);
        return true;
    }
    if (anAttribute->GetName() == "angle_stop") {
        anAttribute->CopyTo(fObject, &KGShellArcSegmentSurface::AngleStop);
        return true;
    }
    if (anAttribute->GetName() == "shell_mesh_count") {
        anAttribute->CopyTo(fObject, &KGShellArcSegmentSurface::ShellMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "shell_mesh_power") {
        anAttribute->CopyTo(fObject, &KGShellArcSegmentSurface::ShellMeshPower);
        return true;
    }
    return false;
}

template<> inline bool KGShellArcSegmentSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "arc_segment") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarArcSegment::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
