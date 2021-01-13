#ifndef KGCYLINDERSURFACEBUILDER_HH_
#define KGCYLINDERSURFACEBUILDER_HH_

#include "KComplexElement.hh"
#include "KGCylinderSurface.hh"
using namespace KGeoBag;

namespace katrin
{

typedef KComplexElement<KGCylinderSurface> KGCylinderSurfaceBuilder;

template<> inline bool KGCylinderSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (anAttribute->GetName() == "z1") {
        anAttribute->CopyTo(fObject, &KGCylinderSurface::Z1);
        return true;
    }
    if (anAttribute->GetName() == "z2") {
        anAttribute->CopyTo(fObject, &KGCylinderSurface::Z2);
        return true;
    }
    if (anAttribute->GetName() == "length") {
        const double tLength = anAttribute->AsReference<double>();
        const double tZMin = tLength / -2.0;
        const double tZMax = tLength / 2.0;
        fObject->Z1(tZMin);
        fObject->Z2(tZMax);
        return true;
    }
    if (anAttribute->GetName() == "r") {
        anAttribute->CopyTo(fObject, &KGCylinderSurface::R);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_count") {
        anAttribute->CopyTo(fObject, &KGCylinderSurface::LongitudinalMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_power") {
        anAttribute->CopyTo(fObject, &KGCylinderSurface::LongitudinalMeshPower);
        return true;
    }
    if (anAttribute->GetName() == "axial_mesh_count") {
        anAttribute->CopyTo(fObject, &KGCylinderSurface::AxialMeshCount);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
