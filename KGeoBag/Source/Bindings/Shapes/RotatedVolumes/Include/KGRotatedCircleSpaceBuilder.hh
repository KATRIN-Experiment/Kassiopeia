#ifndef KGROTATEDCIRCLESPACEBUILDER_HH_
#define KGROTATEDCIRCLESPACEBUILDER_HH_

#include "KGPlanarCircleBuilder.hh"
#include "KGRotatedCircleSpace.hh"

namespace katrin
{

typedef KComplexElement<KGRotatedCircleSpace> KGRotatedCircleSpaceBuilder;

template<> inline bool KGRotatedCircleSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGRotatedCircleSpace::SetName);
        return true;
    }
    if (anAttribute->GetName() == "rotated_mesh_count") {
        anAttribute->CopyTo(fObject, &KGRotatedCircleSpace::RotatedMeshCount);
        return true;
    }
    return false;
}

template<> inline bool KGRotatedCircleSpaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "circle") {
        anElement->CopyTo(fObject->Path().operator->(), &KGPlanarCircle::CopyFrom);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
