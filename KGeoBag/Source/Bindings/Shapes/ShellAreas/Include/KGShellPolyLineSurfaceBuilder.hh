#ifndef KGSHELLPOLYLINESURFACEBUILDER_HH_
#define KGSHELLPOLYLINESURFACEBUILDER_HH_

#include "KGPlanarPolyLineBuilder.hh"
#include "KGShellPolyLineSurface.hh"

namespace katrin
{

//! This produces an angularly limited portion of a surface of revolution from a poly-line that is rotated around the local
//! x-axis.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_shell_poly_line_surface_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: Yes.
//! @example
//! <shell_poly_line_surface name="shell_poly_line_forward_surface" angle_start="270" angle_stop="120" shell_mesh_count="64" shell_mesh_power="6">
//!     <poly_line>
//!         <start_point x="0.2" y="0.3"/>
//!         <next_line x="0.1" y="0.1" line_mesh_count="36" line_mesh_power="4.5"/>
//!         <next_arc x="-0.1" y="0.1" radius="0.2" right="true" short="true" arc_mesh_count="96"/>
//!         <next_line x="-0.2" y="0.2" line_mesh_count="36" line_mesh_power="1."/>
//!         <next_line x="-0.3" y="0.1" line_mesh_count="24" line_mesh_power="4.5"/>
//!     </poly_line>
//! </shell_poly_line_surface>
typedef KComplexElement<KGShellPolyLineSurface> KGShellPolyLineSurfaceBuilder;

template<> inline bool KGShellPolyLineSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGShellPolyLineSurface::SetName);
        return true;
    }
    if (anAttribute->GetName() == "shell_mesh_count") {
        anAttribute->CopyTo(fObject, &KGShellPolyLineSurface::ShellMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "angle_start") {
        anAttribute->CopyTo(fObject, &KGShellPolyLineSurface::AngleStart);
        return true;
    }
    if (anAttribute->GetName() == "angle_stop") {
        anAttribute->CopyTo(fObject, &KGShellPolyLineSurface::AngleStop);
        return true;
    }
    if (anAttribute->GetName() == "shell_mesh_power") {
        anAttribute->CopyTo(fObject, &KGShellPolyLineSurface::ShellMeshPower);
        return true;
    }
    return false;
}

template<> inline bool KGShellPolyLineSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "poly_line") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarPolyLine::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
