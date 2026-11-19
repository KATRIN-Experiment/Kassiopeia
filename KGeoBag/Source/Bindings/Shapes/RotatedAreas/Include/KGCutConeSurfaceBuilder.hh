#ifndef KGCUTCONESURFACEBUILDER_HH_
#define KGCUTCONESURFACEBUILDER_HH_

#include "KComplexElement.hh"
#include "KGCutConeSurface.hh"
using namespace KGeoBag;

namespace katrin
{

//! Produces a truncated cone centered on the local z-axis.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_cut_cone_surface_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: Yes.
//! @example <cut_cone_surface name="cut_cone_surface" z1="0.5" r1="0.6" z2="-0.1" r2="0.2" longitudinal_mesh_count="23" longitudinal_mesh_power="4." axial_mesh_count="48"/>
typedef KComplexElement<KGCutConeSurface> KGCutConeSurfaceBuilder;

template<> inline bool KGCutConeSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (anAttribute->GetName() == "z1") {
        anAttribute->CopyTo(fObject, &KGCutConeSurface::Z1);
        return true;
    }
    if (anAttribute->GetName() == "r1") {
        anAttribute->CopyTo(fObject, &KGCutConeSurface::R1);
        return true;
    }
    if (anAttribute->GetName() == "z2") {
        anAttribute->CopyTo(fObject, &KGCutConeSurface::Z2);
        return true;
    }
    if (anAttribute->GetName() == "r2") {
        anAttribute->CopyTo(fObject, &KGCutConeSurface::R2);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_count") {
        anAttribute->CopyTo(fObject, &KGCutConeSurface::LongitudinalMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_power") {
        anAttribute->CopyTo(fObject, &KGCutConeSurface::LongitudinalMeshPower);
        return true;
    }
    if (anAttribute->GetName() == "axial_mesh_count") {
        anAttribute->CopyTo(fObject, &KGCutConeSurface::AxialMeshCount);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
