#ifndef KGTORUSSURFACEBUILDER_HH_
#define KGTORUSSURFACEBUILDER_HH_

#include "KComplexElement.hh"
#include "KGTorusSurface.hh"
using namespace KGeoBag;

namespace katrin
{

typedef KComplexElement<KGTorusSurface> KGTorusSurfaceBuilder;

template<> bool KGTorusSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (anAttribute->GetName() == "z") {
        anAttribute->CopyTo(fObject, &KGTorusSurface::Z);
        return true;
    }
    if (anAttribute->GetName() == "r") {
        anAttribute->CopyTo(fObject, &KGTorusSurface::R);
        return true;
    }
    if (anAttribute->GetName() == "radius") {
        anAttribute->CopyTo(fObject, &KGTorusSurface::Radius);
        return true;
    }
    if (anAttribute->GetName() == "toroidal_mesh_count") {
        anAttribute->CopyTo(fObject, &KGTorusSurface::ToroidalMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "axial_mesh_count") {
        anAttribute->CopyTo(fObject, &KGTorusSurface::AxialMeshCount);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
