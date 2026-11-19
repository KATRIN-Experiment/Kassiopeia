#ifndef KGBEAMBUILDER_HH_
#define KGBEAMBUILDER_HH_

#include "KComplexElement.hh"
#include "KGBeam.hh"
#include "KGWrappedSpace.hh"
#include "KGWrappedSurface.hh"

using namespace KGeoBag;

namespace katrin
{
struct KGBeamLine
{
    double x1;
    double y1;
    double z1;
    double x2;
    double y2;
    double z2;
};

typedef KComplexElement<KGBeamLine> KGBeamLineBuilder;

template<> inline bool KGBeamLineBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "x1") {
        anAttribute->CopyTo(fObject->x1);
        return true;
    }
    if (anAttribute->GetName() == "y1") {
        anAttribute->CopyTo(fObject->y1);
        return true;
    }
    if (anAttribute->GetName() == "z1") {
        anAttribute->CopyTo(fObject->z1);
        return true;
    }
    if (anAttribute->GetName() == "x2") {
        anAttribute->CopyTo(fObject->x2);
        return true;
    }
    if (anAttribute->GetName() == "y2") {
        anAttribute->CopyTo(fObject->y2);
        return true;
    }
    if (anAttribute->GetName() == "z2") {
        anAttribute->CopyTo(fObject->z2);
        return true;
    }
    return false;
}

using KGBeamBuilder = KComplexElement<KGBeam>;

template<> inline bool KGBeamBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "longitudinal_mesh_count") {
        anAttribute->CopyTo(fObject, &KGBeam::SetNDiscLong);
        return true;
    }
    if (anAttribute->GetName() == "axial_mesh_count") {
        anAttribute->CopyTo(fObject, &KGBeam::SetNDiscRad);
        return true;
    }
    return false;
}

template<> inline bool KGBeamBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "start_line") {
        auto* startLine = anElement->AsPointer<KGBeamLine>();
        double p1[3] = {startLine->x1, startLine->y1, startLine->z1};
        double p2[3] = {startLine->x2, startLine->y2, startLine->z2};
        fObject->AddStartLine(p1, p2);
        return true;
    }
    if (anElement->GetName() == "end_line") {
        auto* endLine = anElement->AsPointer<KGBeamLine>();
        double p1[3] = {endLine->x1, endLine->y1, endLine->z1};
        double p2[3] = {endLine->x2, endLine->y2, endLine->z2};
        fObject->AddEndLine(p1, p2);
        return true;
    }
    return false;
}

//! A beam creates a pipe which connects two polygons at each end. The polygons must have the same number of vertices but
//! may lie in planes which are not parallel to each other. For the description of the parameters involved see the image
//! below.
//!
//! @image Documentation/gh-pages/source/_images/beam_description.png 250pt
//!
//! In the following example, the variable ``[poly]`` dictates the the number of vertexes in each polygon at the ends of the
//! beam.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_beam_surface_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: No.
//! @example
//! <beam_surface name="example_beam_surface">
//!     <beam longitudinal_mesh_count="20" axial_mesh_count="20">
//!         <define name="poly" value="50"/>
//!         <define name="radius" value=".5"/>
//!         <define name="z_start_max" value="-.25"/>
//!         <define name="z_start_min" value="-.75"/>
//!         <define name="z_end_max" value=".75"/>
//!         <define name="z_end_min" value=".25"/>
//!         <loop variable="i" start="0" end="[poly]" step="1">
//!         <start_line x1="{[radius]*TMath::Cos(2.*TMath::Pi()*[i]/[poly])}" y1="{[radius]*TMath::Sin(2.*TMath::Pi()*[i]/[poly])}" z1="{([z_start_max]+[z_start_min])*.5 + ([z_start_max]-[z_start_min])*TMath::Cos(2.*TMath::Pi()*[i]/[poly])}" x2="{[radius]*TMath::Cos(2.*TMath::Pi()*(([i]+1) mod [poly])/[poly])}" y2="{[radius]*TMath::Sin(2.*TMath::Pi()*(([i]+1) mod [poly])/[poly])}" z2="{([z_start_max]+[z_start_min])*.5 + ([z_start_max]-[z_start_min])*TMath::Cos(2.*TMath::Pi()*(([i]+1) mod [poly])/[poly])}"/>
//!             <end_line x1="{[radius]*TMath::Cos(2.*TMath::Pi()*[i]/[poly])}" y1="{[radius]*TMath::Sin(2.*TMath::Pi()*[i]/[poly])}" z1="{([z_end_max]+[z_end_min])*.5 + ([z_end_max]-[z_end_min])*TMath::Cos(2.*TMath::Pi()*[i]/[poly])}" x2="{[radius]*TMath::Cos(2.*TMath::Pi()*(([i]+1) mod [poly])/[poly])}" y2="{[radius]*TMath::Sin(2.*TMath::Pi()*(([i]+1) mod [poly])/[poly])}" z2="{([z_end_max]+[z_end_min])*.5 + ([z_end_max]-[z_end_min])*TMath::Cos(2.*TMath::Pi()*(([i]+1) mod [poly])/[poly])}"/>
//!         </loop>
//!     </beam>
//! </beam_surface>
using KGBeamSurfaceBuilder = KComplexElement<KGWrappedSurface<KGBeam>>;

template<> inline bool KGBeamSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGWrappedSurface<KGBeam>::SetName);
        return true;
    }
    return false;
}

template<> inline bool KGBeamSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "beam") {
        KGBeam* object = nullptr;
        anElement->ReleaseTo(object);
        object->Initialize();
        std::shared_ptr<KGBeam> smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
}


//! For visualization, see :ref:`<beam_surface> <binding-kgwrappedsurface-kgbeam>`.
//!
//! - Works with GeometryViewer: No.
//! - Works with MeshViewer: No.
//! @example
//! <beam_space name="example_beam_space">
//!     <beam longitudinal_mesh_count="20" axial_mesh_count="20">
//!         <define name="poly_" value="50"/>
//!         <define name="radius_" value=".5"/>
//!         <define name="z_start_max_" value="-.25"/>
//!         <define name="z_start_min_" value="-.75"/>
//!         <define name="z_end_max_" value=".75"/>
//!         <define name="z_end_min_" value=".25"/>
//!         <loop variable="i" start="0" end="[poly_]" step="1">
//!             <start_line x1="{[radius_]*TMath::Cos(2.*TMath::Pi()*[i]/[poly_])}" y1="{[radius_]*TMath::Sin(2.*TMath::Pi()*[i]/[poly_])}" z1="{([z_start_max_]+[z_start_min_])*.5 + ([z_start_max_]-[z_start_min_])*TMath::Cos(2.*TMath::Pi()*[i]/[poly_])}" x2="{[radius_]*TMath::Cos(2.*TMath::Pi()*(([i]+1) mod [poly_])/[poly_])}" y2="{[radius_]*TMath::Sin(2.*TMath::Pi()*(([i]+1) mod [poly_])/[poly_])}" z2="{([z_start_max_]+[z_start_min_])*.5 + ([z_start_max_]-[z_start_min_])*TMath::Cos(2.*TMath::Pi()*(([i]+1) mod [poly_])/[poly_])}"/>
//!             <end_line x1="{[radius_]*TMath::Cos(2.*TMath::Pi()*[i]/[poly_])}" y1="{[radius_]*TMath::Sin(2.*TMath::Pi()*[i]/[poly_])}" z1="{([z_end_max_]+[z_end_min_])*.5 + ([z_end_max_]-[z_end_min_])*TMath::Cos(2.*TMath::Pi()*[i]/[poly_])}" x2="{[radius_]*TMath::Cos(2.*TMath::Pi()*(([i]+1) mod [poly_])/[poly_])}" y2="{[radius_]*TMath::Sin(2.*TMath::Pi()*(([i]+1) mod [poly_])/[poly_])}" z2="{([z_end_max_]+[z_end_min_])*.5 + ([z_end_max_]-[z_end_min_])*TMath::Cos(2.*TMath::Pi()*(([i]+1) mod [poly_])/[poly_])}"/>
//!         </loop>
//!     </beam>
//! </beam_space>
using KGBeamSpaceBuilder = KComplexElement<KGWrappedSpace<KGBeam>>;

template<> inline bool KGBeamSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGWrappedSpace<KGBeam>::SetName);
        return true;
    }
    return false;
}

template<> inline bool KGBeamSpaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "beam") {
        KGBeam* object = nullptr;
        anElement->ReleaseTo(object);
        object->Initialize();
        std::shared_ptr<KGBeam> smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
