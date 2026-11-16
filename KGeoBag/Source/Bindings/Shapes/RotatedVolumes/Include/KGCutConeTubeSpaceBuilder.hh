#ifndef KGCUTCONETUBESPACEBUILDER_HH_
#define KGCUTCONETUBESPACEBUILDER_HH_

#include "KComplexElement.hh"
#include "KGCutConeTubeSpace.hh"
using namespace KGeoBag;

namespace katrin
{

//! Generates a tubular frustrum (a cut cone with central axially symmetric void).
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_cut_cone_tube_space_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: Yes.
//! @example <cut_cone_tube_space name="cut_cone_tube_space" z1="-0.3" z2="0.3" r11="0.3" r12="0.48" r21="0.1" r22="0.18" longitudinal_mesh_count="32" longitudinal_mesh_power="2." radial_mesh_count="24" radial_mesh_power="1.5" axial_mesh_count="32"/>
typedef KComplexElement<KGCutConeTubeSpace> KGCutConeTubeSpaceBuilder;

template<> inline bool KGCutConeTubeSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (anAttribute->GetName() == "z1") {
        anAttribute->CopyTo(fObject, &KGCutConeTubeSpace::Z1);
        return true;
    }
    if (anAttribute->GetName() == "z2") {
        anAttribute->CopyTo(fObject, &KGCutConeTubeSpace::Z2);
        return true;
    }
    if (anAttribute->GetName() == "r11") {
        anAttribute->CopyTo(fObject, &KGCutConeTubeSpace::R11);
        return true;
    }
    if (anAttribute->GetName() == "r12") {
        anAttribute->CopyTo(fObject, &KGCutConeTubeSpace::R12);
        return true;
    }
    if (anAttribute->GetName() == "r21") {
        anAttribute->CopyTo(fObject, &KGCutConeTubeSpace::R21);
        return true;
    }
    if (anAttribute->GetName() == "r22") {
        anAttribute->CopyTo(fObject, &KGCutConeTubeSpace::R22);
        return true;
    }
    if (anAttribute->GetName() == "radial_mesh_count") {
        anAttribute->CopyTo(fObject, &KGCutConeTubeSpace::RadialMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "radial_mesh_power") {
        anAttribute->CopyTo(fObject, &KGCutConeTubeSpace::RadialMeshPower);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_count") {
        anAttribute->CopyTo(fObject, &KGCutConeTubeSpace::LongitudinalMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_power") {
        anAttribute->CopyTo(fObject, &KGCutConeTubeSpace::LongitudinalMeshPower);
        return true;
    }
    if (anAttribute->GetName() == "axial_mesh_count") {
        anAttribute->CopyTo(fObject, &KGCutConeTubeSpace::AxialMeshCount);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
