#ifndef KGCIRCULARWIREPINSBUILDER_HH_
#define KGCIRCULARWIREPINSBUILDER_HH_

#include "KComplexElement.hh"
#include "KGCircularWirePins.hh"
#include "KGWrappedSpace.hh"
#include "KGWrappedSurface.hh"

using namespace KGeoBag;

namespace katrin
{
typedef KComplexElement<KGCircularWirePins> KGCircularWirePinsBuilder;

template<> inline bool KGCircularWirePinsBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "inner_radius") {
        anAttribute->CopyTo(fObject, &KGCircularWirePins::SetR1);
        return true;
    }
    if (anAttribute->GetName() == "outer_radius") {
        anAttribute->CopyTo(fObject, &KGCircularWirePins::SetR2);
        return true;
    }
    if (anAttribute->GetName() == "n_pins") {
        anAttribute->CopyTo(fObject, &KGCircularWirePins::SetNPins);
        return true;
    }
    if (anAttribute->GetName() == "diameter") {
        anAttribute->CopyTo(fObject, &KGCircularWirePins::SetDiameter);
        return true;
    }
    if (anAttribute->GetName() == "rotation_angle") {
        anAttribute->CopyTo(fObject, &KGCircularWirePins::SetRotationAngle);
        return true;
    }
    if (anAttribute->GetName() == "mesh_power") {
        anAttribute->CopyTo(fObject, &KGCircularWirePins::SetNDiscPower);
        return true;
    }
    if (anAttribute->GetName() == "mesh_count") {
        anAttribute->CopyTo(fObject, &KGCircularWirePins::SetNDisc);
        return true;
    }

    return false;
}

using KGCircularWirePinsSurfaceBuilder = KComplexElement<KGWrappedSurface<KGCircularWirePins>>;

template<> inline bool KGCircularWirePinsSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGWrappedSurface<KGCircularWirePins>::SetName);
        return true;
    }
    return false;
}

template<> inline bool KGCircularWirePinsSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "circular_wire_pins") {
        KGCircularWirePins* object = nullptr;
        anElement->ReleaseTo(object);
        object->Initialize();
        std::shared_ptr<KGCircularWirePins> smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
}


using KGCircularWirePinsSpaceBuilder = KComplexElement<KGWrappedSpace<KGCircularWirePins>>;

template<> inline bool KGCircularWirePinsSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGWrappedSpace<KGCircularWirePins>::SetName);
        return true;
    }
    return false;
}

template<> inline bool KGCircularWirePinsSpaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "circular_wire_pins") {
        KGCircularWirePins* object = nullptr;
        anElement->ReleaseTo(object);
        object->Initialize();
        std::shared_ptr<KGCircularWirePins> smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
