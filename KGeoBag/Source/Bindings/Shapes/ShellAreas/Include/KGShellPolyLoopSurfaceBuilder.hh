#ifndef KGSHELLPOLYLOOPSURFACEBUILDER_HH_
#define KGSHELLPOLYLOOPSURFACEBUILDER_HH_

#include "KGPlanarPolyLoopBuilder.hh"
#include "KGShellPolyLoopSurface.hh"

namespace katrin
{

//! This produces an angularly limited portion of a surface of revolution from a poly-loop that is rotated around the local
//! x-axis.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_shell_poly_loop_surface_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: Yes.
//! @example
//! <shell_poly_loop_surface name="shell_poly_loop_surface"  angle_start="30" angle_stop="360" shell_mesh_count="64" shell_mesh_power="6">
//!     <poly_loop>
//!         <start_point x="0.3" y="0.3"/>
//!         <next_line x="0.3" y="0.5" line_mesh_count="64" line_mesh_power="2.5"/>
//!         <next_arc x="0.1" y="0.7" radius="0.25" right="false" short="true" arc_mesh_count="64"/>
//!         <next_line x="-0.1" y="0.7" line_mesh_count="64" line_mesh_power="2.5"/>
//!         <next_arc x="-0.3" y="0.5" radius="0.25" right="false" short="true" arc_mesh_count="64"/>
//!         <next_line x="-0.3" y="0.3" line_mesh_count="64" line_mesh_power="2.5"/>
//!         <next_arc x="-0.1" y="0.1" radius="0.25" right="false" short="true" arc_mesh_count="64"/>
//!         <next_line x="0.1" y="0.1" line_mesh_count="64" line_mesh_power="2.5"/>
//!         <last_arc radius="0.25" right="false" short="true" arc_mesh_count="64"/>
//!     </poly_loop>
//! </shell_poly_loop_surface>
typedef KComplexElement<KGShellPolyLoopSurface> KGShellPolyLoopSurfaceBuilder;

template<> inline bool KGShellPolyLoopSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGShellPolyLoopSurface::SetName);
        return true;
    }
    if (anAttribute->GetName() == "angle_start") {
        anAttribute->CopyTo(fObject, &KGShellPolyLoopSurface::AngleStart);
        return true;
    }
    if (anAttribute->GetName() == "angle_stop") {
        anAttribute->CopyTo(fObject, &KGShellPolyLoopSurface::AngleStop);
        return true;
    }
    if (anAttribute->GetName() == "shell_mesh_count") {
        anAttribute->CopyTo(fObject, &KGShellPolyLoopSurface::ShellMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "shell_mesh_power") {
        anAttribute->CopyTo(fObject, &KGShellPolyLoopSurface::ShellMeshPower);
        return true;
    }
    return false;
}

template<> inline bool KGShellPolyLoopSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "poly_loop") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarPolyLoop::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
