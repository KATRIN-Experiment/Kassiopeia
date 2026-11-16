#ifndef KGTORUSSPACEBUILDER_HH_
#define KGTORUSSPACEBUILDER_HH_

#include "KComplexElement.hh"
#include "KGTorusSpace.hh"
using namespace KGeoBag;

namespace katrin
{

//! Produces a torus centered on the local z axis.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_torus_space_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: Yes.
//! @example <torus_space name="torus_space" z="0.2" r="0.5" radius="0.35" toroidal_mesh_count="256" axial_mesh_count="512"/>
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
