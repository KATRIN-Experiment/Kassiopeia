#ifndef KGROTATEDPOLYLINESURFACEBUILDER_HH_
#define KGROTATEDPOLYLINESURFACEBUILDER_HH_

#include "KGPlanarPolyLineBuilder.hh"
#include "KGRotatedPolyLineSurface.hh"

namespace katrin
{

//! This type generates the surface of revolution created by rotating a poly-line around the local x-axis.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_rotated_poly_line_surface_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: Yes.
//! @example
//! <rotated_poly_line_surface name="rotated_poly_line_forward_surface" rotated_mesh_count="64">
//!     <poly_line>
//!         <start_point x="0.2" y="0.3"/>
//!         <next_line x="0.1" y="0.1" line_mesh_count="36" line_mesh_power="4.5"/>
//!         <next_arc x="-0.1" y="0.1" radius="0.2" right="true" short="true" arc_mesh_count="96"/>
//!         <next_line x="-0.2" y="0.2" line_mesh_count="36" line_mesh_power="1."/>
//!         <next_line x="-0.3" y="0.1" line_mesh_count="24" line_mesh_power="4.5"/>
//!     </poly_line>
//! </rotated_poly_line_surface>
typedef KComplexElement<KGRotatedPolyLineSurface> KGRotatedPolyLineSurfaceBuilder;

template<> inline bool KGRotatedPolyLineSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGRotatedPolyLineSurface::SetName);
        return true;
    }
    if (anAttribute->GetName() == "rotated_mesh_count") {
        anAttribute->CopyTo(fObject, &KGRotatedPolyLineSurface::RotatedMeshCount);
        return true;
    }
    return false;
}

template<> inline bool KGRotatedPolyLineSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "poly_line") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarPolyLine::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
