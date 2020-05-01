#ifndef KGDISKSURFACEBUILDER_HH_
#define KGDISKSURFACEBUILDER_HH_

#include "KComplexElement.hh"
#include "KGDiskSurface.hh"

namespace katrin
{

typedef KComplexElement<KGeoBag::KGDiskSurface> KGDiskSurfaceBuilder;

template<> inline bool KGDiskSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    using namespace KGeoBag;

    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (anAttribute->GetName() == "z") {
        anAttribute->CopyTo(fObject, &KGDiskSurface::Z);
        return true;
    }
    if (anAttribute->GetName() == "r") {
        anAttribute->CopyTo(fObject, &KGDiskSurface::R);
        return true;
    }
    if (anAttribute->GetName() == "radial_mesh_count") {
        anAttribute->CopyTo(fObject, &KGDiskSurface::RadialMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "radial_mesh_power") {
        anAttribute->CopyTo(fObject, &KGDiskSurface::RadialMeshPower);
        return true;
    }
    if (anAttribute->GetName() == "axial_mesh_count") {
        anAttribute->CopyTo(fObject, &KGDiskSurface::AxialMeshCount);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
