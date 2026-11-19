#ifndef KGROTATEDPOLYLOOPSPACEBUILDER_HH_
#define KGROTATEDPOLYLOOPSPACEBUILDER_HH_

#include "KGPlanarPolyLoopBuilder.hh"
#include "KGRotatedPolyLoopSpace.hh"

namespace katrin
{

//! Generates a volume enclosed by a surface of revolution produced from rotating a poly-loop
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_rotated_poly_loop_space_model.png 400pt
//!
//! - Works with GeometryViewer: No.
//! - Works with MeshViewer: Yes.
//! @example
//! <rotated_poly_loop_space name="rotated_poly_loop_space" rotated_mesh_count="64">
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
//! </rotated_poly_loop_space>
typedef KComplexElement<KGRotatedPolyLoopSpace> KGRotatedPolyLoopSpaceBuilder;

template<> inline bool KGRotatedPolyLoopSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGRotatedPolyLoopSpace::SetName);
        return true;
    }
    if (anAttribute->GetName() == "rotated_mesh_count") {
        anAttribute->CopyTo(fObject, &KGRotatedPolyLoopSpace::RotatedMeshCount);
        return true;
    }
    return false;
}

template<> inline bool KGRotatedPolyLoopSpaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "poly_loop") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarPolyLoop::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
