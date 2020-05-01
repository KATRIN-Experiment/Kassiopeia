#ifndef KGCYLINDERTUBESPACEBUILDER_HH_
#define KGCYLINDERTUBESPACEBUILDER_HH_

#include "KComplexElement.hh"
#include "KGCylinderTubeSpace.hh"
using namespace KGeoBag;

namespace katrin
{

typedef KComplexElement<KGCylinderTubeSpace> KGCylinderTubeSpaceBuilder;

template<> inline bool KGCylinderTubeSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (anAttribute->GetName() == "z1") {
        anAttribute->CopyTo(fObject, &KGCylinderTubeSpace::Z1);
        return true;
    }
    if (anAttribute->GetName() == "z2") {
        anAttribute->CopyTo(fObject, &KGCylinderTubeSpace::Z2);
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
    if (anAttribute->GetName() == "r1") {
        anAttribute->CopyTo(fObject, &KGCylinderTubeSpace::R1);
        return true;
    }
    if (anAttribute->GetName() == "r2") {
        anAttribute->CopyTo(fObject, &KGCylinderTubeSpace::R2);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_count") {
        anAttribute->CopyTo(fObject, &KGCylinderTubeSpace::LongitudinalMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_power") {
        anAttribute->CopyTo(fObject, &KGCylinderTubeSpace::LongitudinalMeshPower);
        return true;
    }
    if (anAttribute->GetName() == "radial_mesh_count") {
        anAttribute->CopyTo(fObject, &KGCylinderTubeSpace::RadialMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "radial_mesh_power") {
        anAttribute->CopyTo(fObject, &KGCylinderTubeSpace::RadialMeshPower);
        return true;
    }
    if (anAttribute->GetName() == "axial_mesh_count") {
        anAttribute->CopyTo(fObject, &KGCylinderTubeSpace::AxialMeshCount);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif
