#ifndef KGEXTRUDEDCIRCLESURFACEBUILDER_HH_
#define KGEXTRUDEDCIRCLESURFACEBUILDER_HH_

#include "KGExtrudedCircleSurface.hh"
#include "KGPlanarCircleBuilder.hh"

namespace katrin
{

//! This generates the surfaced produced by extruding a circle (this is the same as a cylinder).
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_extruded_circle_space_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: Yes.
//! @example
//! <extruded_circle_surface name="extruded_circle_surface" zmin="-0.1" zmax="0.1" extruded_mesh_count="32" extruded_mesh_power="1">
//!     <circle x="0.1" y="0.5" radius="0.3" circle_mesh_count="128"/>
//! </extruded_circle_surface>
typedef KComplexElement<KGExtrudedCircleSurface> KGExtrudedCircleSurfaceBuilder;

template<> inline bool KGExtrudedCircleSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGExtrudedCircleSurface::SetName);
        return true;
    }
    if (anAttribute->GetName() == "zmin") {
        anAttribute->CopyTo(fObject, &KGExtrudedCircleSurface::ZMin);
        return true;
    }
    if (anAttribute->GetName() == "zmax") {
        anAttribute->CopyTo(fObject, &KGExtrudedCircleSurface::ZMax);
        return true;
    }
    if (anAttribute->GetName() == "extruded_mesh_count") {
        anAttribute->CopyTo(fObject, &KGExtrudedCircleSurface::ExtrudedMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "extruded_mesh_power") {
        anAttribute->CopyTo(fObject, &KGExtrudedCircleSurface::ExtrudedMeshPower);
        return true;
    }
    return false;
}

template<> inline bool KGExtrudedCircleSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "circle") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarCircle::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
