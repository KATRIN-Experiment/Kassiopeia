#ifndef KGRODBUILDER_HH_
#define KGRODBUILDER_HH_

#include "KComplexElement.hh"
#include "KGRod.hh"
#include "KGWrappedSpace.hh"
#include "KGWrappedSurface.hh"

using namespace KGeoBag;

namespace katrin
{
struct KGRodVertex
{
    double x;
    double y;
    double z;
};

typedef KComplexElement<KGRodVertex> KGRodVertexBuilder;

template<> inline bool KGRodVertexBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "x") {
        anAttribute->CopyTo(fObject->x);
        return true;
    }
    if (anAttribute->GetName() == "y") {
        anAttribute->CopyTo(fObject->y);
        return true;
    }
    if (anAttribute->GetName() == "z") {
        anAttribute->CopyTo(fObject->z);
        return true;
    }
    return false;
}

typedef KComplexElement<KGRod> KGRodBuilder;

template<> inline bool KGRodBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGRod::SetName);
        return true;
    }
    if (anAttribute->GetName() == "radius") {
        anAttribute->CopyTo(fObject, &KGRod::SetRadius);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_count") {
        anAttribute->CopyTo(fObject, &KGRod::SetNDiscLong);
        return true;
    }
    if (anAttribute->GetName() == "axial_mesh_count") {
        anAttribute->CopyTo(fObject, &KGRod::SetNDiscRad);
        return true;
    }
    return false;
}

template<> inline bool KGRodBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "vertex") {
        auto* vtx = anElement->AsPointer<KGRodVertex>();
        double p[3] = {vtx->x, vtx->y, vtx->z};
        fObject->AddPoint(p);
        return true;
    }
    return false;
}

typedef KComplexElement<KGWrappedSurface<KGRod>> KGRodSurfaceBuilder;

template<> inline bool KGRodSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGWrappedSurface<KGRod>::SetName);
        return true;
    }
    return false;
}

template<> inline bool KGRodSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "rod") {
        KGRod* object = nullptr;
        anElement->ReleaseTo(object);
        object->Initialize();
        std::shared_ptr<KGRod> smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
}


typedef KComplexElement<KGWrappedSpace<KGRod>> KGRodSpaceBuilder;

template<> inline bool KGRodSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGWrappedSpace<KGRod>::SetName);
        return true;
    }
    return false;
}

template<> inline bool KGRodSpaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "rod") {
        KGRod* object = nullptr;
        anElement->ReleaseTo(object);
        object->Initialize();
        std::shared_ptr<KGRod> smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
