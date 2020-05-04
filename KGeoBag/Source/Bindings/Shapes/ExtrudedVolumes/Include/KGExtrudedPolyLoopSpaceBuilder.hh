#ifndef KGEXTRUDEDPOLYLOOPSPACEBUILDER_HH_
#define KGEXTRUDEDPOLYLOOPSPACEBUILDER_HH_

#include "KGExtrudedPolyLoopSpace.hh"
#include "KGPlanarPolyLoopBuilder.hh"

namespace katrin
{

typedef KComplexElement<KGExtrudedPolyLoopSpace> KGExtrudedPolyLoopSpaceBuilder;

template<> inline bool KGExtrudedPolyLoopSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSpace::SetName);
        return true;
    }
    if (anAttribute->GetName() == "zmin") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSpace::ZMin);
        return true;
    }
    if (anAttribute->GetName() == "zmax") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSpace::ZMax);
        return true;
    }
    if (anAttribute->GetName() == "extruded_mesh_count") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSpace::ExtrudedMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "extruded_mesh_power") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSpace::ExtrudedMeshPower);
        return true;
    }
    if (anAttribute->GetName() == "flattened_mesh_count") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSpace::FlattenedMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "flattened_mesh_power") {
        anAttribute->CopyTo(fObject, &KGExtrudedPolyLoopSpace::FlattenedMeshPower);
        return true;
    }
    return false;
}

template<> inline bool KGExtrudedPolyLoopSpaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "poly_loop") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarPolyLoop::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
