#ifndef KGLINEARWIREGRIDBUILDER_HH_
#define KGLINEARWIREGRIDBUILDER_HH_

#include "KComplexElement.hh"
#include "KGLinearWireGrid.hh"
#include "KGWrappedSpace.hh"
#include "KGWrappedSurface.hh"

using namespace KGeoBag;

namespace katrin
{
typedef KComplexElement<KGLinearWireGrid> KGLinearWireGridBuilder;

template<> inline bool KGLinearWireGridBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "radius") {
        anAttribute->CopyTo(fObject, &KGLinearWireGrid::SetR);
        return true;
    }
    if (anAttribute->GetName() == "pitch") {
        anAttribute->CopyTo(fObject, &KGLinearWireGrid::SetPitch);
        return true;
    }
    if (anAttribute->GetName() == "diameter") {
        anAttribute->CopyTo(fObject, &KGLinearWireGrid::SetDiameter);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_count") {
        anAttribute->CopyTo(fObject, &KGLinearWireGrid::SetNDisc);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_power") {
        anAttribute->CopyTo(fObject, &KGLinearWireGrid::SetNDiscPower);
        return true;
    }
    if (anAttribute->GetName() == "add_outer_circle") {
        anAttribute->CopyTo(fObject, &KGLinearWireGrid::SetOuterCircle);
        return true;
    }
    return false;
}

typedef KComplexElement<KGWrappedSurface<KGLinearWireGrid>> KGLinearWireGridSurfaceBuilder;

template<> inline bool KGLinearWireGridSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGWrappedSurface<KGLinearWireGrid>::SetName);
        return true;
    }
    return false;
}

template<> inline bool KGLinearWireGridSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "linear_wire_grid") {
        KGLinearWireGrid* object = nullptr;
        anElement->ReleaseTo(object);
        object->Initialize();
        std::shared_ptr<KGLinearWireGrid> smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
}


typedef KComplexElement<KGWrappedSpace<KGLinearWireGrid>> KGLinearWireGridSpaceBuilder;

template<> inline bool KGLinearWireGridSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGWrappedSpace<KGLinearWireGrid>::SetName);
        return true;
    }
    return false;
}

template<> inline bool KGLinearWireGridSpaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "linear_wire_grid") {
        KGLinearWireGrid* object = nullptr;
        anElement->ReleaseTo(object);
        object->Initialize();
        std::shared_ptr<KGLinearWireGrid> smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
