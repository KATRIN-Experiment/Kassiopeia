#ifndef KGROTATEDCIRCLESURFACEBUILDER_HH_
#define KGROTATEDCIRCLESURFACEBUILDER_HH_

#include "KGPlanarCircleBuilder.hh"
#include "KGRotatedCircleSurface.hh"

namespace katrin
{

//! This type generates the surface of revolution created by rotating a circle around the local x-axis. This shape is
//! essentially or torus, or a section of a torus.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_rotated_circle_surface_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: Yes.
//! @example
//! <rotated_circle_surface name="rotated_circle_surface" rotated_mesh_count="32">
//!     <circle x="0.1" y="0.5" radius="0.3" circle_mesh_count="32"/>
//! </rotated_circle_surface>
typedef KComplexElement<KGRotatedCircleSurface> KGRotatedCircleSurfaceBuilder;

template<> inline bool KGRotatedCircleSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGRotatedCircleSurface::SetName);
        return true;
    }
    if (anAttribute->GetName() == "rotated_mesh_count") {
        anAttribute->CopyTo(fObject, &KGRotatedCircleSurface::RotatedMeshCount);
        return true;
    }
    return false;
}

template<> inline bool KGRotatedCircleSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "circle") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarCircle::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
