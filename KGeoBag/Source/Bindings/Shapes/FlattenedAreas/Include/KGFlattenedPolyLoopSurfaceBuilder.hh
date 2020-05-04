#ifndef KGFLATTENEDPOLYLINESURFACEBUILDER_HH_
#define KGFLATTENEDPOLYLINESURFACEBUILDER_HH_

#include "KGFlattenedPolyLoopSurface.hh"
#include "KGPlanarPolyLoopBuilder.hh"

namespace katrin
{

typedef KComplexElement<KGFlattenedPolyLoopSurface> KGFlattenedPolyLoopSurfaceBuilder;

template<> inline bool KGFlattenedPolyLoopSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGFlattenedPolyLoopSurface::SetName);
        return true;
    }
    if (anAttribute->GetName() == "z") {
        anAttribute->CopyTo(fObject, &KGFlattenedPolyLoopSurface::Z);
        return true;
    }
    if (anAttribute->GetName() == "flattened_mesh_count") {
        anAttribute->CopyTo(fObject, &KGFlattenedPolyLoopSurface::FlattenedMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "flattened_mesh_power") {
        anAttribute->CopyTo(fObject, &KGFlattenedPolyLoopSurface::FlattenedMeshPower);
        return true;
    }
    return false;
}

template<> inline bool KGFlattenedPolyLoopSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "poly_loop") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarPolyLoop::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
