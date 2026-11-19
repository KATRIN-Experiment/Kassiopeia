#ifndef KGBOXSPACEBUILDER_HH_
#define KGBOXSPACEBUILDER_HH_

#include "KComplexElement.hh"
#include "KGBoxSpace.hh"
using namespace KGeoBag;

namespace katrin
{

//! Produces an axis aligned rectangular prism.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_box_space_model.png 400pt
//!
//! - Works with MeshViewer: No.
//! - Works with GeometryViewer: Yes.
//! @example <box_space name="box_space" xa="0.1" xb="0.65" x_mesh_count="1000" x_mesh_power="5.5" ya="-0.4" yb="0.3" y_mesh_count="100" y_mesh_power="1.8" za="0.4" zb="-0.1" z_mesh_count="10" z_mesh_power="1."/>
typedef KComplexElement<KGBoxSpace> KGBoxSpaceBuilder;

template<> inline bool KGBoxSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (anAttribute->GetName() == "xa") {
        anAttribute->CopyTo(fObject, &KGBoxSpace::XA);
        return true;
    }
    if (anAttribute->GetName() == "xb") {
        anAttribute->CopyTo(fObject, &KGBoxSpace::XB);
        return true;
    }
    if (anAttribute->GetName() == "ya") {
        anAttribute->CopyTo(fObject, &KGBoxSpace::YA);
        return true;
    }
    if (anAttribute->GetName() == "yb") {
        anAttribute->CopyTo(fObject, &KGBoxSpace::YB);
        return true;
    }
    if (anAttribute->GetName() == "za") {
        anAttribute->CopyTo(fObject, &KGBoxSpace::ZA);
        return true;
    }
    if (anAttribute->GetName() == "zb") {
        anAttribute->CopyTo(fObject, &KGBoxSpace::ZB);
        return true;
    }
    if (anAttribute->GetName() == "x_mesh_count") {
        anAttribute->CopyTo(fObject, &KGBoxSpace::XMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "x_mesh_power") {
        anAttribute->CopyTo(fObject, &KGBoxSpace::XMeshPower);
        return true;
    }
    if (anAttribute->GetName() == "y_mesh_count") {
        anAttribute->CopyTo(fObject, &KGBoxSpace::YMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "y_mesh_power") {
        anAttribute->CopyTo(fObject, &KGBoxSpace::YMeshPower);
        return true;
    }
    if (anAttribute->GetName() == "z_mesh_count") {
        anAttribute->CopyTo(fObject, &KGBoxSpace::ZMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "z_mesh_power") {
        anAttribute->CopyTo(fObject, &KGBoxSpace::ZMeshPower);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
