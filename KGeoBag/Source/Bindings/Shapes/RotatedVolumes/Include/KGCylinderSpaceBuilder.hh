#ifndef KGCYLINDERSPACEBUILDER_HH_
#define KGCYLINDERSPACEBUILDER_HH_

#include "KComplexElement.hh"
#include "KGCylinderSpace.hh"
using namespace KGeoBag;

namespace katrin
{

//! Produces a cylinder space centered on the local z axis.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_cylinder_space_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: Yes.
//!
//! An XML example is as follows:
//!
//! .. code-block:: xml
//!
//!     <cylinder_space name="cylinder_space" z1="-0.4" z2="0.4" r="0.3" longitudinal_mesh_count="32" longitudinal_mesh_power="2." radial_mesh_count="24" radial_mesh_power="1.5" axial_mesh_count="32"/>
//!
//! The parameters are:
//!     - z1: the first z coordinate in meters
//!     - z2: the second z coordinate in meters
//!     - r: the radius in meters
//!     - longitudinal_mesh_count: longitudinal mesh parameter (default is 1)
//!     - longitudinal_mesh_power: longitudinal meshing power (default is 1.)
//!     - radial_mesh_count: radial mesh parameter (default is 1)
//!     - radial_mesh_power: radial meshing power (default is 1.)
//!     - axial_mesh_count: axial mesh parameter (default is 16)
typedef KComplexElement<KGCylinderSpace> KGCylinderSpaceBuilder;

template<> inline bool KGCylinderSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (anAttribute->GetName() == "z1") {
        anAttribute->CopyTo(fObject, &KGCylinderSpace::Z1);
        return true;
    }
    if (anAttribute->GetName() == "z2") {
        anAttribute->CopyTo(fObject, &KGCylinderSpace::Z2);
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
    if (anAttribute->GetName() == "r") {
        anAttribute->CopyTo(fObject, &KGCylinderSpace::R);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_count") {
        anAttribute->CopyTo(fObject, &KGCylinderSpace::LongitudinalMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_power") {
        anAttribute->CopyTo(fObject, &KGCylinderSpace::LongitudinalMeshPower);
        return true;
    }
    if (anAttribute->GetName() == "radial_mesh_count") {
        anAttribute->CopyTo(fObject, &KGCylinderSpace::RadialMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "radial_mesh_power") {
        anAttribute->CopyTo(fObject, &KGCylinderSpace::RadialMeshPower);
        return true;
    }
    if (anAttribute->GetName() == "axial_mesh_count") {
        anAttribute->CopyTo(fObject, &KGCylinderSpace::AxialMeshCount);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
