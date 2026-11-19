#ifndef KGFLATTENEDCIRCLESURFACEBUILDER_HH_
#define KGFLATTENEDCIRCLESURFACEBUILDER_HH_

#include "KGFlattenedCircleSurface.hh"
#include "KGPlanarCircleBuilder.hh"

namespace katrin
{

//! A flattened circle surface is just a surface with circular shape, where (x,y,z) are the coordinates for the center of
//! the circle and r is the radius:
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_flattened_circle_surface_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: Yes.
//! @example
//! <flattened_circle_surface name="flattened_circle_surface" z="0." flattened_mesh_count="10" flattened_mesh_power="4.">
//!     <circle x="0.1" y="0.2" radius="0.5" circle_mesh_count="128"/>
//! </flattened_circle_surface>
typedef KComplexElement<KGFlattenedCircleSurface> KGFlattenedCircleSurfaceBuilder;

template<> inline bool KGFlattenedCircleSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGFlattenedCircleSurface::SetName);
        return true;
    }
    if (anAttribute->GetName() == "z") {
        anAttribute->CopyTo(fObject, &KGFlattenedCircleSurface::Z);
        return true;
    }
    if (anAttribute->GetName() == "flattened_mesh_count") {
        anAttribute->CopyTo(fObject, &KGFlattenedCircleSurface::FlattenedMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "flattened_mesh_power") {
        anAttribute->CopyTo(fObject, &KGFlattenedCircleSurface::FlattenedMeshPower);
        return true;
    }
    return false;
}

template<> inline bool KGFlattenedCircleSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "circle") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarCircle::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
