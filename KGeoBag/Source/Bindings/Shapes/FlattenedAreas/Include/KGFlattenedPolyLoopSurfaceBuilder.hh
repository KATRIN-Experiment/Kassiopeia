#ifndef KGFLATTENEDPOLYLINESURFACEBUILDER_HH_
#define KGFLATTENEDPOLYLINESURFACEBUILDER_HH_

#include "KGFlattenedPolyLoopSurface.hh"
#include "KGPlanarPolyLoopBuilder.hh"

namespace katrin
{

//! A flattened poly loop surface is a surface consisting of several lines, arcs or both, creating a polygon of your desired
//! shape. The first line and the last line are connected automatically. By using the features of the
//! :ref:`<poly_loop> <binding-kgplanarpolyloop>`, this provides a very flexible method of designing surface elements.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_flattened_poly_loop_surface_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: Yes.
//! @example
//! <flattened_poly_loop_surface name="flattened_poly_loop_surface" z="0.1" flattened_mesh_count="10" flattened_mesh_power="4.">
//!     <poly_loop>
//!         <start_point x="0." y="0."/>
//!         <next_line x="0.4" y="-0.1" line_mesh_count="10" line_mesh_power="2.5"/>
//!         <next_arc x="-0.1" y="0.4" radius="0.6" right="true" short="false" arc_mesh_count="256"/>
//!         <last_line line_mesh_count="10" line_mesh_power="2.5"/>
//!     </poly_loop>
//! </flattened_poly_loop_surface>
typedef KComplexElement<KGFlattenedPolyLoopSurface> KGFlattenedPolyLoopSurfaceBuilder;

template<> inline bool KGFlattenedPolyLoopSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGFlattenedPolyLoopSurface::SetName);
        return true;
    }
    if (anAttribute->GetName() == "z") {
        anAttribute->CopyTo(fObject, &KGFlattenedPolyLoopSurface::Z);
        return true;
    }
    if (anAttribute->GetName() == "flattened_mesh_count") {
        anAttribute->CopyTo(fObject, &KGFlattenedPolyLoopSurface::FlattenedMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "flattened_mesh_power") {
        anAttribute->CopyTo(fObject, &KGFlattenedPolyLoopSurface::FlattenedMeshPower);
        return true;
    }
    return false;
}

template<> inline bool KGFlattenedPolyLoopSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "poly_loop") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarPolyLoop::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
