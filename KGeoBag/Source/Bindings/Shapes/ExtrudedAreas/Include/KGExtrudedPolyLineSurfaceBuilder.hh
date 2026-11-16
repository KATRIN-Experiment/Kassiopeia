#ifndef KGEXTRUDEDPOLYLINESURFACEBUILDER_HH_
#define KGEXTRUDEDPOLYLINESURFACEBUILDER_HH_

#include "KGExtrudedPolyLineSurface.hh"
#include "KGPlanarPolyLineBuilder.hh"

namespace katrin
{

//! This generates a surface by extruding a poly-line.
//! @example
//! <extruded_poly_line_surface name="extruded_poly_line_surface" zmin="-0.3" zmax="0.2" extruded_mesh_count="96" extruded_mesh_power="6.3">
//!     <poly_line>
//!         <start_point x="-0.3" y="0.1"/>
//!         <next_line x="-0.2" y="0.2" line_mesh_count="24" line_mesh_power="4.5"/>
//!         <next_line x="-0.1" y="0.1" line_mesh_count="36" line_mesh_power="1."/>
//!         <next_arc x="0.1" y="0.1" radius="0.2" right="false" short="true" arc_mesh_count="96"/>
//!         <next_line x="0.2" y="0.3" line_mesh_count="36" line_mesh_power="4.5"/>
//!     </poly_line>
//! </extruded_poly_line_surface>
typedef KComplexElement<KGExtrudedPolyLineSurface> KGExtrudedPolyLineSurfaceBuilder;

template<> inline bool KGExtrudedPolyLineSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLineSurface::SetName);
        return true;
    }
    if (anAttribute->GetName() == "zmin") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLineSurface::ZMin);
        return true;
    }
    if (anAttribute->GetName() == "zmax") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLineSurface::ZMax);
        return true;
    }
    if (anAttribute->GetName() == "extruded_mesh_count") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLineSurface::ExtrudedMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "extruded_mesh_power") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLineSurface::ExtrudedMeshPower);
        return true;
    }
    return false;
}

template<> inline bool KGExtrudedPolyLineSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "poly_line") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarPolyLine::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
