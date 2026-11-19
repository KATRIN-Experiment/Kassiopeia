#ifndef KGEXTRUDEDPOLYLOOPSURFACEBUILDER_HH_
#define KGEXTRUDEDPOLYLOOPSURFACEBUILDER_HH_

#include "KGExtrudedPolyLoopSurface.hh"
#include "KGPlanarPolyLoopBuilder.hh"

namespace katrin
{

//! This generates a surface by extruding a poly-loop.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_extruded_poly_loop_surface_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: Yes.
//! @example
//! <extruded_poly_loop_surface name="extruded_poly_loop_surface" zmin="-0.3" zmax="0.3" extruded_mesh_count="37" extruded_mesh_power="6.3">
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
//! </extruded_poly_loop_surface>
typedef KComplexElement<KGExtrudedPolyLoopSurface> KGExtrudedPolyLoopSurfaceBuilder;

template<> inline bool KGExtrudedPolyLoopSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSurface::SetName);
        return true;
    }
    if (anAttribute->GetName() == "zmin") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSurface::ZMin);
        return true;
    }
    if (anAttribute->GetName() == "zmax") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSurface::ZMax);
        return true;
    }
    if (anAttribute->GetName() == "extruded_mesh_count") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSurface::ExtrudedMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "extruded_mesh_power") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSurface::ExtrudedMeshPower);
        return true;
    }
    return false;
}

template<> inline bool KGExtrudedPolyLoopSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "poly_loop") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarPolyLoop::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
