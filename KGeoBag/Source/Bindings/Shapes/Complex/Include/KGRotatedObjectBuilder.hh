#ifndef KGROTATEDOBJECTBUILDER_HH_
#define KGROTATEDOBJECTBUILDER_HH_

#include "KComplexElement.hh"
#include "KGRotatedObject.hh"
#include "KGWrappedSpace.hh"
#include "KGWrappedSurface.hh"

#include <memory>

using namespace KGeoBag;

namespace katrin
{
typedef KComplexElement<KGRotatedObject::Line> KGRotatedObjectLineBuilder;

template<> inline bool KGRotatedObjectLineBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "z1") {
        double p1[2] = {0., 0.};
        anAttribute->CopyTo(p1[0]);
        p1[1] = fObject->GetP1(1);
        fObject->SetP1(p1);
        return true;
    }
    if (anAttribute->GetName() == "r1") {
        double p1[2] = {0., 0.};
        p1[0] = fObject->GetP1(0);
        anAttribute->CopyTo(p1[1]);
        fObject->SetP1(p1);
        return true;
    }
    if (anAttribute->GetName() == "z2") {
        double p2[2] = {0., 0.};
        anAttribute->CopyTo(p2[0]);
        p2[1] = fObject->GetP2(1);
        fObject->SetP2(p2);
        return true;
    }
    if (anAttribute->GetName() == "r2") {
        double p2[2] = {0., 0.};
        p2[0] = fObject->GetP2(0);
        anAttribute->CopyTo(p2[1]);
        fObject->SetP2(p2);
        return true;
    }
    return false;
}

typedef KComplexElement<KGRotatedObject::Arc> KGRotatedObjectArcBuilder;

template<> inline bool KGRotatedObjectArcBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "z1") {
        double p1[2] = {0., 0.};
        anAttribute->CopyTo(p1[0]);
        p1[1] = fObject->GetP1(1);
        fObject->SetP1(p1);
        return true;
    }
    if (anAttribute->GetName() == "r1") {
        double p1[2] = {0., 0.};
        p1[0] = fObject->GetP1(0);
        anAttribute->CopyTo(p1[1]);
        fObject->SetP1(p1);
        return true;
    }
    if (anAttribute->GetName() == "z2") {
        double p2[2] = {0., 0.};
        anAttribute->CopyTo(p2[0]);
        p2[1] = fObject->GetP2(1);
        fObject->SetP2(p2);
        return true;
    }
    if (anAttribute->GetName() == "r2") {
        double p2[2] = {0., 0.};
        p2[0] = fObject->GetP2(0);
        anAttribute->CopyTo(p2[1]);
        fObject->SetP2(p2);
        return true;
    }
    if (anAttribute->GetName() == "radius") {
        anAttribute->CopyTo(fObject, &KGRotatedObject::Arc::SetRadius);
        return true;
    }
    if (anAttribute->GetName() == "positive_orientation") {
        anAttribute->CopyTo(fObject, &KGRotatedObject::Arc::SetOrientation);
        return true;
    }
    return false;
}

typedef KComplexElement<KGRotatedObject> KGRotatedObjectBuilder;

template<> inline bool KGRotatedObjectBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "longitudinal_mesh_count_start") {
        anAttribute->CopyTo(fObject, &KGRotatedObject::SetNPolyBegin);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_count_end") {
        anAttribute->CopyTo(fObject, &KGRotatedObject::SetNPolyEnd);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_count") {
        anAttribute->CopyTo(fObject, &KGRotatedObject::SetNPolyBegin);
        anAttribute->CopyTo(fObject, &KGRotatedObject::SetNPolyEnd);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_power") {
        anAttribute->CopyTo(fObject, &KGRotatedObject::SetDiscretizationPower);
        return true;
    }
    return false;
}

template<> inline bool KGRotatedObjectBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "line") {
        anElement->AsPointer<KGRotatedObject::Line>()->Initialize();
        anElement->ReleaseTo(fObject, &KGRotatedObject::AddSegment);
        return true;
    }
    if (anElement->GetName() == "arc") {
        anElement->AsPointer<KGRotatedObject::Arc>()->Initialize();
        anElement->ReleaseTo(fObject, &KGRotatedObject::AddSegment);
        return true;
    }
    return false;
}

typedef KComplexElement<KGWrappedSurface<KGRotatedObject>> KGRotatedSurfaceBuilder;

template<> inline bool KGRotatedSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGWrappedSurface<KGRotatedObject>::SetName);
        return true;
    }
    return false;
}

template<> inline bool KGRotatedSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "rotated_object") {
        KGRotatedObject* object = nullptr;
        anElement->ReleaseTo(object);
        object->Initialize();
        std::shared_ptr<KGRotatedObject> smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
}


typedef KComplexElement<KGWrappedSpace<KGRotatedObject>> KGRotatedSpaceBuilder;

template<> inline bool KGRotatedSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGWrappedSpace<KGRotatedObject>::SetName);
        return true;
    }
    return false;
}

template<> inline bool KGRotatedSpaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "rotated_object") {
        KGRotatedObject* object = nullptr;
        anElement->ReleaseTo(object);
        object->Initialize();
        std::shared_ptr<KGRotatedObject> smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
