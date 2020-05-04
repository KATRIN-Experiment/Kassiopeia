#ifndef KGROTATEDPOLYLOOPSURFACEBUILDER_HH_
#define KGROTATEDPOLYLOOPSURFACEBUILDER_HH_

#include "KGPlanarPolyLoopBuilder.hh"
#include "KGRotatedPolyLoopSurface.hh"

namespace katrin
{

typedef KComplexElement<KGRotatedPolyLoopSurface> KGRotatedPolyLoopSurfaceBuilder;

template<> inline bool KGRotatedPolyLoopSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGRotatedPolyLoopSurface::SetName);
        return true;
    }
    if (anAttribute->GetName() == "rotated_mesh_count") {
        anAttribute->CopyTo(fObject, &KGRotatedPolyLoopSurface::RotatedMeshCount);
        return true;
    }
    return false;
}

template<> inline bool KGRotatedPolyLoopSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "poly_loop") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarPolyLoop::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
