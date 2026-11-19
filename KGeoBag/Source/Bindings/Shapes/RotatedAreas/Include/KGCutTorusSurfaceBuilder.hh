#ifndef KGCUTTORUSSURFACEBUILDER_HH_
#define KGCUTTORUSSURFACEBUILDER_HH_

#include "KComplexElement.hh"
#include "KGCutTorusSurface.hh"

namespace katrin
{

//! Produces an angularly limited toroidal section centered on the z axis.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_cut_torus_surface_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: Yes.
//! @example <cut_torus_surface name="cut_torus_surface" z1="0.3" r1="0.1" z2="-0.1" r2="0.2" radius="0.3" right="true" short="true" toroidal_mesh_count="128" axial_mesh_count="256"/>
typedef KComplexElement<KGeoBag::KGCutTorusSurface> KGCutTorusSurfaceBuilder;

template<> inline bool KGCutTorusSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    using namespace KGeoBag;

    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (anAttribute->GetName() == "z1") {
        anAttribute->CopyTo(fObject, &KGCutTorusSurface::Z1);
        return true;
    }
    if (anAttribute->GetName() == "r1") {
        anAttribute->CopyTo(fObject, &KGCutTorusSurface::R1);
        return true;
    }
    if (anAttribute->GetName() == "z2") {
        anAttribute->CopyTo(fObject, &KGCutTorusSurface::Z2);
        return true;
    }
    if (anAttribute->GetName() == "r2") {
        anAttribute->CopyTo(fObject, &KGCutTorusSurface::R2);
        return true;
    }
    if (anAttribute->GetName() == "radius") {
        anAttribute->CopyTo(fObject, &KGCutTorusSurface::Radius);
        return true;
    }
    if (anAttribute->GetName() == "right") {
        anAttribute->CopyTo(fObject, &KGCutTorusSurface::Right);
        return true;
    }
    if (anAttribute->GetName() == "short") {
        anAttribute->CopyTo(fObject, &KGCutTorusSurface::Right);
        return true;
    }
    if (anAttribute->GetName() == "toroidal_mesh_count") {
        anAttribute->CopyTo(fObject, &KGCutTorusSurface::ToroidalMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "axial_mesh_count") {
        anAttribute->CopyTo(fObject, &KGCutTorusSurface::AxialMeshCount);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
