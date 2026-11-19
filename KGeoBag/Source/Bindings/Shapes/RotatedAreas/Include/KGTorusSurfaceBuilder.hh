#ifndef KGTORUSSURFACEBUILDER_HH_
#define KGTORUSSURFACEBUILDER_HH_

#include "KComplexElement.hh"
#include "KGTorusSurface.hh"
using namespace KGeoBag;

namespace katrin
{

//! Generates a torus centered on the local z axis.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_torus_surface_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: Yes.
//! @example <torus_surface name="torus_surface" z="0.2" r="0.5" radius="0.35" toroidal_mesh_count="256" axial_mesh_count="512"/>
typedef KComplexElement<KGTorusSurface> KGTorusSurfaceBuilder;

template<> inline bool KGTorusSurfaceBuilder::AddAttribute(KContainer* anAttribute)
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
