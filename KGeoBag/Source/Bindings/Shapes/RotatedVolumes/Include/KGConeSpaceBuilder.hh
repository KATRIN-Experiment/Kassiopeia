#ifndef KGCONESPACEBUILDER_HH_
#define KGCONESPACEBUILDER_HH_

#include "KComplexElement.hh"
#include "KGConeSpace.hh"
using namespace KGeoBag;

namespace katrin
{

//! Generates a conical space centered on the local z axis.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_cone_space_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: Yes.
//! @example <cone_space name="cone_space" za="-0.1" zb="0.65" rb="0.4" longitudinal_mesh_count="28" longitudinal_mesh_power="1.8" radial_mesh_count="56" radial_mesh_power="1." axial_mesh_count="24"/>
typedef KComplexElement<KGConeSpace> KGConeSpaceBuilder;

template<> inline bool KGConeSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (anAttribute->GetName() == "za") {
        anAttribute->CopyTo(fObject, &KGConeSpace::ZA);
        return true;
    }
    if (anAttribute->GetName() == "zb") {
        anAttribute->CopyTo(fObject, &KGConeSpace::ZB);
        return true;
    }
    if (anAttribute->GetName() == "rb") {
        anAttribute->CopyTo(fObject, &KGConeSpace::RB);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_count") {
        anAttribute->CopyTo(fObject, &KGConeSpace::LongitudinalMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_power") {
        anAttribute->CopyTo(fObject, &KGConeSpace::LongitudinalMeshPower);
        return true;
    }
    if (anAttribute->GetName() == "radial_mesh_count") {
        anAttribute->CopyTo(fObject, &KGConeSpace::RadialMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "radial_mesh_power") {
        anAttribute->CopyTo(fObject, &KGConeSpace::RadialMeshPower);
        return true;
    }
    if (anAttribute->GetName() == "axial_mesh_count") {
        anAttribute->CopyTo(fObject, &KGConeSpace::AxialMeshCount);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
