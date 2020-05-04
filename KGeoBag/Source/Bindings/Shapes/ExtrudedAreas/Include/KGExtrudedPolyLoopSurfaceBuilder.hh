#ifndef KGEXTRUDEDPOLYLOOPSURFACEBUILDER_HH_
#define KGEXTRUDEDPOLYLOOPSURFACEBUILDER_HH_

#include "KGExtrudedPolyLoopSurface.hh"
#include "KGPlanarPolyLoopBuilder.hh"

namespace katrin
{

typedef KComplexElement<KGExtrudedPolyLoopSurface> KGExtrudedPolyLoopSurfaceBuilder;

template<> inline bool KGExtrudedPolyLoopSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSurface::SetName);
        return true;
    }
    if (anAttribute->GetName() == "zmin") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSurface::ZMin);
        return true;
    }
    if (anAttribute->GetName() == "zmax") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSurface::ZMax);
        return true;
    }
    if (anAttribute->GetName() == "extruded_mesh_count") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSurface::ExtrudedMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "extruded_mesh_power") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSurface::ExtrudedMeshPower);
        return true;
    }
    return false;
}

template<> inline bool KGExtrudedPolyLoopSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "poly_loop") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarPolyLoop::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
