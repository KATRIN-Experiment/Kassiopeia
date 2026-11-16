#ifndef KGROTATEDPOLYLOOPSURFACEBUILDER_HH_
#define KGROTATEDPOLYLOOPSURFACEBUILDER_HH_

#include "KGPlanarPolyLoopBuilder.hh"
#include "KGRotatedPolyLoopSurface.hh"

namespace katrin
{

//! This type generates the surface of revolution created by rotating a poly-loop around the local x-axis.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_rotated_poly_loop_surface_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: Yes.
//! @example
//! <rotated_poly_loop_surface name="rotated_poly_loop_surface" rotated_mesh_count="64">
//!     <poly_loop>
//!         <start_point x="0.3" y="0.3"/>
//!         <next_line x="0.3" y="0.5" line_mesh_count="36" line_mesh_power="2.5"/>
//!         <next_arc x="0.1" y="0.7" radius="0.25" right="false" short="true" arc_mesh_count="64"/>
//!         <next_line x="-0.1" y="0.7" line_mesh_count="36" line_mesh_power="2.5"/>
//!         <next_arc x="-0.3" y="0.5" radius="0.25" right="false" short="true" arc_mesh_count="64"/>
//!         <next_line x="-0.3" y="0.3" line_mesh_count="36" line_mesh_power="2.5"/>
//!         <next_arc x="-0.1" y="0.1" radius="0.25" right="false" short="true" arc_mesh_count="64"/>
//!         <next_line x="0.1" y="0.1" line_mesh_count="36" line_mesh_power="2.5"/>
//!         <last_arc radius="0.25" right="false" short="true" arc_mesh_count="64"/>
//!     </poly_loop>
//! </rotated_poly_loop_surface>
typedef KComplexElement<KGRotatedPolyLoopSurface> KGRotatedPolyLoopSurfaceBuilder;

template<> inline bool KGRotatedPolyLoopSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGRotatedPolyLoopSurface::SetName);
        return true;
    }
    if (anAttribute->GetName() == "rotated_mesh_count") {
        anAttribute->CopyTo(fObject, &KGRotatedPolyLoopSurface::RotatedMeshCount);
        return true;
    }
    return false;
}

template<> inline bool KGRotatedPolyLoopSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "poly_loop") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarPolyLoop::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
