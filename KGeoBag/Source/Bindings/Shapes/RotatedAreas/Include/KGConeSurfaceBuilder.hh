#ifndef KGCONESURFACEBUILDER_HH_
#define KGCONESURFACEBUILDER_HH_

#include "KComplexElement.hh"
#include "KGConeSurface.hh"
using namespace KGeoBag;

namespace katrin
{

typedef KComplexElement<KGConeSurface> KGConeSurfaceBuilder;

template<> inline bool KGConeSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (anAttribute->GetName() == "za") {
        anAttribute->CopyTo(fObject, &KGConeSurface::ZA);
        return true;
    }
    if (anAttribute->GetName() == "zb") {
        anAttribute->CopyTo(fObject, &KGConeSurface::ZB);
        return true;
    }
    if (anAttribute->GetName() == "rb") {
        anAttribute->CopyTo(fObject, &KGConeSurface::RB);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_count") {
        anAttribute->CopyTo(fObject, &KGConeSurface::LongitudinalMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_power") {
        anAttribute->CopyTo(fObject, &KGConeSurface::LongitudinalMeshPower);
        return true;
    }
    if (anAttribute->GetName() == "axial_mesh_count") {
        anAttribute->CopyTo(fObject, &KGConeSurface::AxialMeshCount);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
