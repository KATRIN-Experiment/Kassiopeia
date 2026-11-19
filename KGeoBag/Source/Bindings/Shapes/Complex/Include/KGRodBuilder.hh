#ifndef KGRODBUILDER_HH_
#define KGRODBUILDER_HH_

#include "KComplexElement.hh"
#include "KGRod.hh"
#include "KGWrappedSpace.hh"
#include "KGWrappedSurface.hh"

using namespace KGeoBag;

namespace katrin
{
struct KGRodVertex
{
    double x;
    double y;
    double z;
};

typedef KComplexElement<KGRodVertex> KGRodVertexBuilder;

template<> inline bool KGRodVertexBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "x") {
        anAttribute->CopyTo(fObject->x);
        return true;
    }
    if (anAttribute->GetName() == "y") {
        anAttribute->CopyTo(fObject->y);
        return true;
    }
    if (anAttribute->GetName() == "z") {
        anAttribute->CopyTo(fObject->z);
        return true;
    }
    return false;
}

using KGRodBuilder = KComplexElement<KGRod>;

template<> inline bool KGRodBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGRod::SetName);
        return true;
    }
    if (anAttribute->GetName() == "radius") {
        anAttribute->CopyTo(fObject, &KGRod::SetRadius);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_count") {
        anAttribute->CopyTo(fObject, &KGRod::SetNDiscLong);
        return true;
    }
    if (anAttribute->GetName() == "axial_mesh_count") {
        anAttribute->CopyTo(fObject, &KGRod::SetNDiscRad);
        return true;
    }
    return false;
}

template<> inline bool KGRodBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "vertex") {
        auto* vtx = anElement->AsPointer<KGRodVertex>();
        double p[3] = {vtx->x, vtx->y, vtx->z};
        fObject->AddPoint(p);
        return true;
    }
    return false;
}

//! The rod element can be used to create a rod like structure along a series of linear segments in three dimensional space.
//! The rod has a circular cross section which is approximated by a polygon (governed by the ``axial_mesh_count``
//! parameter). This shape is defined by a radius (thickness of the rod) and a piecewise linear path specified by a by a
//! list of in-order vertexes.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_rod_surface_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: No.
//!
//! The following XML example creates a helix shaped rod surface, as follows:
//! @example
//! <rod_surface name="example_rod_surface">
//!     <rod radius="0.2" longitudinal_mesh_count="100" axial_mesh_count="12">
//!         <define name="n_turns" value="5"/>
//!         <define name="n_segments_per_turn" value="36"/>
//!         <define name="coil_radius" value="1."/>
//!         <define name="height_per_turn" value="1."/>
//!         <loop variable="i" start="0" end="{[n_turns]*[n_segments_per_turn]}" step="1">
//!             <vertex x="{[coil_radius]*TMath::Cos(2.*TMath::Pi()*([i] mod [n_segments_per_turn])/[n_segments_per_turn])}" y="{[coil_radius]*TMath::Sin(2.*TMath::Pi()*([i] mod [n_segments_per_turn])/[n_segments_per_turn])}" z="{[i]*[height_per_turn]/[n_segments_per_turn]}"/>
//!         </loop>
//!     </rod>
//! </rod_surface>
using KGRodSurfaceBuilder = KComplexElement<KGWrappedSurface<KGRod>>;

template<> inline bool KGRodSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGWrappedSurface<KGRod>::SetName);
        return true;
    }
    return false;
}

template<> inline bool KGRodSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "rod") {
        KGRod* object = nullptr;
        anElement->ReleaseTo(object);
        object->Initialize();
        std::shared_ptr<KGRod> smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
}


//! For visualization, see :ref:`<rod_surface> <binding-kgwrappedsurface-kgrod>`.
//!
//! - Works with GeometryViewer: No.
//! - Works with MeshViewer: No.
//! @example
//! <rod_space name="example_rod_space">
//!     <rod radius="0.2" longitudinal_mesh_count="100" axial_mesh_count="12">
//!         <define name="n_turns_" value="5"/>
//!         <define name="n_segments_per_turn_" value="36"/>
//!         <define name="coil_radius_" value="1."/>
//!         <define name="height_per_turn_" value="1."/>
//!         <loop variable="i" start="0" end="{[n_turns_]*[n_segments_per_turn_]}" step="1">
//!             <vertex x="{[coil_radius_]*TMath::Cos(2.*TMath::Pi()*([i] mod [n_segments_per_turn_])/[n_segments_per_turn_])}" y="{[coil_radius_]*TMath::Sin(2.*TMath::Pi()*([i] mod [n_segments_per_turn_])/[n_segments_per_turn_])}" z="{[i]*[height_per_turn_]/[n_segments_per_turn_]}"/>
//!         </loop>
//!     </rod>
//! </rod_space>
using KGRodSpaceBuilder = KComplexElement<KGWrappedSpace<KGRod>>;

template<> inline bool KGRodSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGWrappedSpace<KGRod>::SetName);
        return true;
    }
    return false;
}

template<> inline bool KGRodSpaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "rod") {
        KGRod* object = nullptr;
        anElement->ReleaseTo(object);
        object->Initialize();
        std::shared_ptr<KGRod> smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
