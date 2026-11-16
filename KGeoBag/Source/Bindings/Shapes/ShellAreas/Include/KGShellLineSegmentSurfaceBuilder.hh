#ifndef KGSHELLLINESEGMENTSURFACEBUILDER_HH_
#define KGSHELLLINESEGMENTSURFACEBUILDER_HH_

#include "KGPlanarLineSegmentBuilder.hh"
#include "KGShellLineSegmentSurface.hh"

namespace katrin
{

//! This produces an angularly limited portion of a surface of revolution from a line that is rotated around the local
//! x-axis.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_shell_line_segment_surface_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: Yes.
//! @example
//! <shell_line_segment_surface name="shell_line_segment_forward_surface" angle_start="240" shell_mesh_count="96" shell_mesh_power="6">
//!     <line_segment x1="0.15" y1="0.1" x2="-0.1" y2="0.05" line_mesh_count="36" line_mesh_power="2.2"/>
//! </shell_line_segment_surface>
typedef KComplexElement<KGShellLineSegmentSurface> KGShellLineSegmentSurfaceBuilder;

template<> inline bool KGShellLineSegmentSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGShellLineSegmentSurface::SetName);
        return true;
    }
    if (anAttribute->GetName() == "angle_start") {
        anAttribute->CopyTo(fObject, &KGShellLineSegmentSurface::AngleStart);
        return true;
    }
    if (anAttribute->GetName() == "angle_stop") {
        anAttribute->CopyTo(fObject, &KGShellLineSegmentSurface::AngleStop);
        return true;
    }
    if (anAttribute->GetName() == "shell_mesh_count") {
        anAttribute->CopyTo(fObject, &KGShellLineSegmentSurface::ShellMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "shell_mesh_power") {
        anAttribute->CopyTo(fObject, &KGShellLineSegmentSurface::ShellMeshPower);
        return true;
    }
    return false;
}

template<> inline bool KGShellLineSegmentSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "line_segment") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarLineSegment::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
