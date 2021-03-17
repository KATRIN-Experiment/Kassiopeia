#ifndef KGTORUSSPACEBUILDER_HH_
#define KGTORUSSPACEBUILDER_HH_

#include "KComplexElement.hh"
#include "KGTorusSpace.hh"
using namespace KGeoBag;

namespace katrin
{

typedef KComplexElement<KGTorusSpace> KGTorusSpaceBuilder;

template<> inline bool KGTorusSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (anAttribute->GetName() == "z") {
        anAttribute->CopyTo(fObject, &KGTorusSpace::Z);
        return true;
    }
    if (anAttribute->GetName() == "r") {
        anAttribute->CopyTo(fObject, &KGTorusSpace::R);
        return true;
    }
    if (anAttribute->GetName() == "radius") {
        anAttribute->CopyTo(fObject, &KGTorusSpace::Radius);
        return true;
    }
    if (anAttribute->GetName() == "toroidal_mesh_count") {
        anAttribute->CopyTo(fObject, &KGTorusSpace::ToroidalMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "axial_mesh_count") {
        anAttribute->CopyTo(fObject, &KGTorusSpace::AxialMeshCount);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
