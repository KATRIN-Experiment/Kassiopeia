#ifndef KGCUTCONESPACEBUILDER_HH_
#define KGCUTCONESPACEBUILDER_HH_

#include "KComplexElement.hh"
#include "KGCutConeSpace.hh"
using namespace KGeoBag;

namespace katrin
{

typedef KComplexElement<KGCutConeSpace> KGCutConeSpaceBuilder;

template<> inline bool KGCutConeSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (anAttribute->GetName() == "z1") {
        anAttribute->CopyTo(fObject, &KGCutConeSpace::Z1);
        return true;
    }
    if (anAttribute->GetName() == "r1") {
        anAttribute->CopyTo(fObject, &KGCutConeSpace::R1);
        return true;
    }
    if (anAttribute->GetName() == "z2") {
        anAttribute->CopyTo(fObject, &KGCutConeSpace::Z2);
        return true;
    }
    if (anAttribute->GetName() == "r2") {
        anAttribute->CopyTo(fObject, &KGCutConeSpace::R2);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_count") {
        anAttribute->CopyTo(fObject, &KGCutConeSpace::LongitudinalMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_power") {
        anAttribute->CopyTo(fObject, &KGCutConeSpace::LongitudinalMeshPower);
        return true;
    }
    if (anAttribute->GetName() == "radial_mesh_count") {
        anAttribute->CopyTo(fObject, &KGCutConeSpace::RadialMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "radial_mesh_power") {
        anAttribute->CopyTo(fObject, &KGCutConeSpace::RadialMeshPower);
        return true;
    }
    if (anAttribute->GetName() == "axial_mesh_count") {
        anAttribute->CopyTo(fObject, &KGCutConeSpace::AxialMeshCount);
        return true;
    }
    return false;
}

}  // namespace katrin


#endif
