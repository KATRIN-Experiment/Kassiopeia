#ifndef KGROTATEDCIRCLESPACEBUILDER_HH_
#define KGROTATEDCIRCLESPACEBUILDER_HH_

#include "KGPlanarCircleBuilder.hh"
#include "KGRotatedCircleSpace.hh"

namespace katrin
{

//! Generates a volume by rotating a circle (torus).
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_rotated_circle_space_model.png 400pt
//!
//! - Works with GeometryViewer: Yes.
//! - Works with MeshViewer: Yes.
//! @example
//! <rotated_circle_space name="rotated_circle_space" rotated_mesh_count="128">
//!     <circle x="0.1" y="0.5" radius="0.3" circle_mesh_count="128"/>
//! </rotated_circle_space>
typedef KComplexElement<KGRotatedCircleSpace> KGRotatedCircleSpaceBuilder;

template<> inline bool KGRotatedCircleSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGRotatedCircleSpace::SetName);
        return true;
    }
    if (anAttribute->GetName() == "rotated_mesh_count") {
        anAttribute->CopyTo(fObject, &KGRotatedCircleSpace::RotatedMeshCount);
        return true;
    }
    return false;
}

template<> inline bool KGRotatedCircleSpaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "circle") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarCircle::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
