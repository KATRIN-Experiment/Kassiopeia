#ifndef KGEXTRUDEDPOLYLINESURFACEBUILDER_HH_
#define KGEXTRUDEDPOLYLINESURFACEBUILDER_HH_

#include "KGExtrudedPolyLineSurface.hh"
#include "KGPlanarPolyLineBuilder.hh"

namespace katrin
{

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
