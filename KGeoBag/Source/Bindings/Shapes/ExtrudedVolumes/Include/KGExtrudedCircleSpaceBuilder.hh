#ifndef KGEXTRUDEDCIRCLESPACEBUILDER_HH_
#define KGEXTRUDEDCIRCLESPACEBUILDER_HH_

#include "KGExtrudedCircleSpace.hh"
#include "KGPlanarCircleBuilder.hh"

namespace katrin
{

//! Generates a volume by extruding a cycle (cynlinder).
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_extruded_circle_space_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: Yes.
//! @example
//! <extruded_circle_space name="extruded_circle_space" zmin="-0.1" zmax="0.1" extruded_mesh_count="32" extruded_mesh_power="1" flattened_mesh_count="28" flattened_mesh_power="1.4">
//!     <circle x="0.1" y="0.5" radius="0.3" circle_mesh_count="128"/>
//! </extruded_circle_space>
typedef KComplexElement<KGExtrudedCircleSpace> KGExtrudedCircleSpaceBuilder;

template<> inline bool KGExtrudedCircleSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGExtrudedCircleSpace::SetName);
        return true;
    }
    if (anAttribute->GetName() == "zmin") {
        anAttribute->CopyTo(fObject, &KGExtrudedCircleSpace::ZMin);
        return true;
    }
    if (anAttribute->GetName() == "zmax") {
        anAttribute->CopyTo(fObject, &KGExtrudedCircleSpace::ZMax);
        return true;
    }
    if (anAttribute->GetName() == "extruded_mesh_count") {
        anAttribute->CopyTo(fObject, &KGExtrudedCircleSpace::ExtrudedMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "extruded_mesh_power") {
        anAttribute->CopyTo(fObject, &KGExtrudedCircleSpace::ExtrudedMeshPower);
        return true;
    }
    if (anAttribute->GetName() == "flattened_mesh_count") {
        anAttribute->CopyTo(fObject, &KGExtrudedCircleSpace::FlattenedMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "flattened_mesh_power") {
        anAttribute->CopyTo(fObject, &KGExtrudedCircleSpace::FlattenedMeshPower);
        return true;
    }
    return false;
}

template<> inline bool KGExtrudedCircleSpaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "circle") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarCircle::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
