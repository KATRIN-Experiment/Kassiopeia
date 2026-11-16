#ifndef KGROTATEDPOLYLINESPACEBUILDER_HH_
#define KGROTATEDPOLYLINESPACEBUILDER_HH_

#include "KGPlanarPolyLineBuilder.hh"
#include "KGRotatedPolyLineSpace.hh"

namespace katrin
{

//! Generates a volume enclosed by a surface of revolution produced from rotating a poly-line.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_rotated_poly_line_space_model.png 400pt
//!
//! - Works with GeometryViewer: Yes.
//! - Works with MeshViewer: Yes.
//! @example
//! <rotated_poly_line_space name="rotated_poly_line_reverse_space" rotated_mesh_count="128" flattened_mesh_count="36" flattened_mesh_power="3.8">
//!     <poly_line>
//!         <start_point x="-0.1" y="0.1"/>
//!         <next_arc x="-0.3" y="0.3" radius="0.315" right="false" short="true" arc_mesh_count="24"/>
//!         <next_line x="0.2" y="0.4" line_mesh_count="52" line_mesh_power="3.5"/>
//!         <next_line x="0.1" y="0.1" line_mesh_count="24" line_mesh_power="2."/>
//!     </poly_line>
//! </rotated_poly_line_space>
typedef KComplexElement<KGRotatedPolyLineSpace> KGRotatedPolyLineSpaceBuilder;

template<> inline bool KGRotatedPolyLineSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGRotatedPolyLineSpace::SetName);
        return true;
    }
    if (anAttribute->GetName() == "rotated_mesh_count") {
        anAttribute->CopyTo(fObject, &KGRotatedPolyLineSpace::RotatedMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "flattened_mesh_count") {
        anAttribute->CopyTo(fObject, &KGRotatedPolyLineSpace::FlattenedMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "flattened_mesh_power") {
        anAttribute->CopyTo(fObject, &KGRotatedPolyLineSpace::FlattenedMeshPower);
        return true;
    }
    return false;
}

template<> inline bool KGRotatedPolyLineSpaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "poly_line") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarPolyLine::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
