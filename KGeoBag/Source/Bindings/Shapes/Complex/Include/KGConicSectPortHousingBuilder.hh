#ifndef KGCONICSECTPORTHOUSINGBUILDER_HH_
#define KGCONICSECTPORTHOUSINGBUILDER_HH_

#include "KComplexElement.hh"
#include "KGConicSectPortHousing.hh"
#include "KGWrappedSpace.hh"
#include "KGWrappedSurface.hh"

using namespace KGeoBag;

namespace katrin
{

typedef KComplexElement<KGConicSectPortHousing::OrthogonalPort> KGConicSectPortHousingOrthogonalPortBuilder;

template<> inline bool KGConicSectPortHousingOrthogonalPortBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "box_radial_mesh_count") {
        fObject->SetXDisc(anAttribute->AsReference<int>());
        return true;
    }
    if (anAttribute->GetName() == "box_curve_mesh_count") {
        fObject->SetAlphaPolySub(anAttribute->AsReference<int>());
        return true;
    }
    if (anAttribute->GetName() == "cylinder_longitudinal_mesh_count") {
        fObject->SetCylDisc(anAttribute->AsReference<int>());
        return true;
    }
    if (anAttribute->GetName() == "cylinder_axial_mesh_count") {
        fObject->SetPolySub(anAttribute->AsReference<int>());
        return true;
    }
    if (anAttribute->GetName() == "radius") {
        fObject->SetRSub(anAttribute->AsReference<double>());
        return true;
    }
    if (anAttribute->GetName() == "x") {
        double a[3] = {};
        anAttribute->CopyTo(a[0]);
        a[1] = fObject->GetASub(1);
        a[2] = fObject->GetASub(2);
        fObject->SetASub(a);
        return true;
    }
    if (anAttribute->GetName() == "y") {
        double a[3] = {};
        a[0] = fObject->GetASub(0);
        anAttribute->CopyTo(a[1]);
        a[2] = fObject->GetASub(2);
        fObject->SetASub(a);
        return true;
    }
    if (anAttribute->GetName() == "z") {
        double a[3] = {};
        a[0] = fObject->GetASub(0);
        a[1] = fObject->GetASub(1);
        anAttribute->CopyTo(a[2]);
        fObject->SetASub(a);
        return true;
    }
    return false;
}

using KGConicSectPortHousingParaxialPortBuilder = KComplexElement<KGConicSectPortHousing::ParaxialPort>;

template<> inline bool KGConicSectPortHousingParaxialPortBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "box_radial_mesh_count") {
        fObject->SetXDisc(anAttribute->AsReference<int>());
        return true;
    }
    if (anAttribute->GetName() == "box_curve_mesh_count") {
        fObject->SetAlphaPolySub(anAttribute->AsReference<int>());
        return true;
    }
    if (anAttribute->GetName() == "cylinder_longitudinal_mesh_count") {
        fObject->SetCylDisc(anAttribute->AsReference<int>());
        return true;
    }
    if (anAttribute->GetName() == "cylinder_axial_mesh_count") {
        fObject->SetPolySub(anAttribute->AsReference<int>());
        return true;
    }
    if (anAttribute->GetName() == "radius") {
        fObject->SetRSub(anAttribute->AsReference<double>());
        return true;
    }
    if (anAttribute->GetName() == "x") {
        double a[3] = {};
        anAttribute->CopyTo(a[0]);
        a[1] = fObject->GetASub(1);
        a[2] = fObject->GetASub(2);
        fObject->SetASub(a);
        return true;
    }
    if (anAttribute->GetName() == "y") {
        double a[3] = {};
        a[0] = fObject->GetASub(0);
        anAttribute->CopyTo(a[1]);
        a[2] = fObject->GetASub(2);
        fObject->SetASub(a);
        return true;
    }
    if (anAttribute->GetName() == "z") {
        double a[3] = {};
        a[0] = fObject->GetASub(0);
        a[1] = fObject->GetASub(1);
        anAttribute->CopyTo(a[2]);
        fObject->SetASub(a);
        return true;
    }
    return false;
}

using KGConicSectPortHousingBuilder = KComplexElement<KGConicSectPortHousing>;

template<> inline bool KGConicSectPortHousingBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "r1") {
        fObject->SetRAMain(anAttribute->AsReference<double>());
        return true;
    }
    if (anAttribute->GetName() == "z1") {
        fObject->SetZAMain(anAttribute->AsReference<double>());
        return true;
    }
    if (anAttribute->GetName() == "r2") {
        fObject->SetRBMain(anAttribute->AsReference<double>());
        return true;
    }
    if (anAttribute->GetName() == "z2") {
        fObject->SetZBMain(anAttribute->AsReference<double>());
        return true;
    }
    if (anAttribute->GetName() == "axial_mesh_count") {
        fObject->SetPolyMain(anAttribute->AsReference<int>());
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_count") {
        fObject->SetNumDiscMain(anAttribute->AsReference<int>());
        return true;
    }
    return false;
}

template<> inline bool KGConicSectPortHousingBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "orthogonal_port") {
        KGConicSectPortHousing::OrthogonalPort* orthogonalPort = nullptr;
        anElement->ReleaseTo(orthogonalPort);
        fObject->AddPort(orthogonalPort);
        orthogonalPort->Initialize();
        return true;
    }
    if (anElement->GetName() == "paraxial_port") {
        KGConicSectPortHousing::ParaxialPort* paraxialPort = nullptr;
        anElement->ReleaseTo(paraxialPort);
        fObject->AddPort(paraxialPort);
        paraxialPort->Initialize();
        return true;
    }
    return false;
}

//! This shape is a cut cone surface with circular ports. The ports can be created either orthogonal to the surface or
//! parallel to the cone axis (paraxial). The coordinates (x,y,z) are the end points of the port.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_conic_section_port_housing_surface_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: No.
//! @example
//! <conic_section_port_housing_surface name="example_conic_section_port_housing_surface">
//!   <conic_section_port_housing r1="2.75" z1="-1.79675" r2="0.25" z2="-0.067" longitudinal_mesh_count="100" axial_mesh_count="100">
//! <define name="offset" value="2."/>
//! <define name="nPorts" value="6"/>
//! <loop variable="i" start="0" end="{[nPorts]-1}" step="1">
//!   <if condition="{([i] mod 2) eq 0}">
//!     <orthogonal_port x="{[offset]*TMath::Cos(2.*TMath::Pi()*[i]/[nPorts])}" y="{[offset]*TMath::Sin(2.*TMath::Pi()*[i]/[nPorts])}" z=".5" radius="{.05 + .05*([i]+1)}"/>
//!   </if>
//!   <if condition="{([i] mod 2) eq 1}">
//!     <paraxial_port x="{[offset]*TMath::Cos(2.*TMath::Pi()*[i]/[nPorts])}" y="{[offset]*TMath::Sin(2.*TMath::Pi()*[i]/[nPorts])}" z=".5" radius="{.05 + .05*([i]+1)}"/>
//!   </if>
//! </loop>
//!   </conic_section_port_housing>
//! </conic_section_port_housing_surface>
using KGConicSectPortHousingSurfaceBuilder = KComplexElement<KGWrappedSurface<KGConicSectPortHousing>>;

template<> inline bool KGConicSectPortHousingSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGWrappedSurface<KGConicSectPortHousing>::SetName);
        return true;
    }
    return false;
}

template<> inline bool KGConicSectPortHousingSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "conic_section_port_housing") {
        KGConicSectPortHousing* object = nullptr;
        anElement->ReleaseTo(object);
        std::shared_ptr<KGConicSectPortHousing> smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
}


//! For visualization, see :ref:`<conic_section_port_housing_surface> <binding-kgwrappedsurface-kgconicsectporthousing>`.
//!
//! - Works with GeometryViewer: No.
//! - Works with MeshViewer: No.
//! @example
//! <conic_section_port_housing_space name="example_conic_section_port_housing_space">
//!     <conic_section_port_housing r1="2.75" z1="-1.79675" r2="0.25" z2="-0.067" longitudinal_mesh_count="100" axial_mesh_count="100">
//!         <define name="offset_" value="2."/>
//!         <define name="nPorts_" value="6"/>
//!         <loop variable="i" start="0" end="{[nPorts_]-1}" step="1">
//!             <if condition="{([i] mod 2) eq 0}">
//!                 <orthogonal_port x="{[offset_]*TMath::Cos(2.*TMath::Pi()*[i]/[nPorts_])}" y="{[offset_]*TMath::Sin(2.*TMath::Pi()*[i]/[nPorts_])}" z=".5" radius="{.05 + .05*([i]+1)}"/>
//!             </if>
//!             <if condition="{([i] mod 2) eq 1}">
//!                 <paraxial_port x="{[offset_]*TMath::Cos(2.*TMath::Pi()*[i]/[nPorts_])}" y="{[offset_]*TMath::Sin(2.*TMath::Pi()*[i]/[nPorts_])}" z=".5" radius="{.05 + .05*([i]+1)}"/>
//!             </if>
//!         </loop>
//!     </conic_section_port_housing>
//! </conic_section_port_housing_space>
using KGConicSectPortHousingSpaceBuilder = KComplexElement<KGWrappedSpace<KGConicSectPortHousing>>;

template<> inline bool KGConicSectPortHousingSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGWrappedSpace<KGConicSectPortHousing>::SetName);
        return true;
    }
    return false;
}

template<> inline bool KGConicSectPortHousingSpaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "conic_section_port_housing") {
        KGConicSectPortHousing* object = nullptr;
        anElement->ReleaseTo(object);
        std::shared_ptr<KGConicSectPortHousing> smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
