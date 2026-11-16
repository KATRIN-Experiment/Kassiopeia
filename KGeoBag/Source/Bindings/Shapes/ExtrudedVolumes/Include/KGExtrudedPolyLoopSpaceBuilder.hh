#ifndef KGEXTRUDEDPOLYLOOPSPACEBUILDER_HH_
#define KGEXTRUDEDPOLYLOOPSPACEBUILDER_HH_

#include "KGExtrudedPolyLoopSpace.hh"
#include "KGPlanarPolyLoopBuilder.hh"

namespace katrin
{

//! Generates a volume by extruding a poly-loop.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_extruded_poly_loop_space_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: Yes.
//! @example
//! <extruded_poly_loop_space name="extruded_poly_loop_space" zmin="-0.7" zmax="0.7" extruded_mesh_count="37" extruded_mesh_power="6.3" flattened_mesh_count="28" flattened_mesh_power="1.4">
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
//! </extruded_poly_loop_space>
typedef KComplexElement<KGExtrudedPolyLoopSpace> KGExtrudedPolyLoopSpaceBuilder;

template<> inline bool KGExtrudedPolyLoopSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSpace::SetName);
        return true;
    }
    if (anAttribute->GetName() == "zmin") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSpace::ZMin);
        return true;
    }
    if (anAttribute->GetName() == "zmax") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSpace::ZMax);
        return true;
    }
    if (anAttribute->GetName() == "extruded_mesh_count") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSpace::ExtrudedMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "extruded_mesh_power") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSpace::ExtrudedMeshPower);
        return true;
    }
    if (anAttribute->GetName() == "flattened_mesh_count") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSpace::FlattenedMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "flattened_mesh_power") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSpace::FlattenedMeshPower);
        return true;
    }
    return false;
}

template<> inline bool KGExtrudedPolyLoopSpaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "poly_loop") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarPolyLoop::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
