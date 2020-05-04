#ifndef KGROTATEDPOLYLOOPSPACEBUILDER_HH_
#define KGROTATEDPOLYLOOPSPACEBUILDER_HH_

#include "KGPlanarPolyLoopBuilder.hh"
#include "KGRotatedPolyLoopSpace.hh"

namespace katrin
{

typedef KComplexElement<KGRotatedPolyLoopSpace> KGRotatedPolyLoopSpaceBuilder;

template<> inline bool KGRotatedPolyLoopSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGRotatedPolyLoopSpace::SetName);
        return true;
    }
    if (anAttribute->GetName() == "rotated_mesh_count") {
        anAttribute->CopyTo(fObject, &KGRotatedPolyLoopSpace::RotatedMeshCount);
        return true;
    }
    return false;
}

template<> inline bool KGRotatedPolyLoopSpaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "poly_loop") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarPolyLoop::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
