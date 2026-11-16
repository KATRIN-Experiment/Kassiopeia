#ifndef KGEXTRUDEDOBJECTBUILDER_HH_
#define KGEXTRUDEDOBJECTBUILDER_HH_

#include "KComplexElement.hh"
#include "KGExtrudedObject.hh"
#include "KGWrappedSpace.hh"
#include "KGWrappedSurface.hh"

using namespace KGeoBag;

namespace katrin
{
typedef KComplexElement<KGExtrudedObject::Line> KGExtrudedObjectLineBuilder;

template<> inline bool KGExtrudedObjectLineBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "x1") {
        double p1[2] = {};
        anAttribute->CopyTo(p1[0]);
        p1[1] = fObject->GetP1(1);
        fObject->SetP1(p1);
        return true;
    }
    if (anAttribute->GetName() == "y1") {
        double p1[2] = {};
        p1[0] = fObject->GetP1(0);
        anAttribute->CopyTo(p1[1]);
        fObject->SetP1(p1);
        return true;
    }
    if (anAttribute->GetName() == "x2") {
        double p2[2] = {};
        anAttribute->CopyTo(p2[0]);
        p2[1] = fObject->GetP2(1);
        fObject->SetP2(p2);
        return true;
    }
    if (anAttribute->GetName() == "y2") {
        double p2[2] = {};
        p2[0] = fObject->GetP2(0);
        anAttribute->CopyTo(p2[1]);
        fObject->SetP2(p2);
        return true;
    }
    return false;
}

using KGExtrudedObjectArcBuilder = KComplexElement<KGExtrudedObject::Arc>;

template<> inline bool KGExtrudedObjectArcBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "x1") {
        double p1[2] = {};
        anAttribute->CopyTo(p1[0]);
        p1[1] = fObject->GetP1(1);
        fObject->SetP1(p1);
        return true;
    }
    if (anAttribute->GetName() == "y1") {
        double p1[2] = {};
        p1[0] = fObject->GetP1(0);
        anAttribute->CopyTo(p1[1]);
        fObject->SetP1(p1);
        return true;
    }
    if (anAttribute->GetName() == "x2") {
        double p2[2] = {};
        anAttribute->CopyTo(p2[0]);
        p2[1] = fObject->GetP2(1);
        fObject->SetP2(p2);
        return true;
    }
    if (anAttribute->GetName() == "y2") {
        double p2[2] = {};
        p2[0] = fObject->GetP2(0);
        anAttribute->CopyTo(p2[1]);
        fObject->SetP2(p2);
        return true;
    }
    if (anAttribute->GetName() == "radius") {
        anAttribute->CopyTo(fObject, &KGExtrudedObject::Arc::SetRadius);
        return true;
    }
    if (anAttribute->GetName() == "positive_orientation") {
        anAttribute->CopyTo(fObject, &KGExtrudedObject::Arc::IsPositivelyOriented);
        return true;
    }
    return false;
}

using KGExtrudedObjectBuilder = KComplexElement<KGExtrudedObject>;

template<> inline bool KGExtrudedObjectBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "z_min") {
        anAttribute->CopyTo(fObject, &KGExtrudedObject::SetZMin);
        return true;
    }
    if (anAttribute->GetName() == "z_max") {
        anAttribute->CopyTo(fObject, &KGExtrudedObject::SetZMax);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_count") {
        anAttribute->CopyTo(fObject, &KGExtrudedObject::SetNDisc);
        return true;
    }
    if (anAttribute->GetName() == "longitudinal_mesh_power") {
        anAttribute->CopyTo(fObject, &KGExtrudedObject::SetDiscretizationPower);
        return true;
    }
    if (anAttribute->GetName() == "extruded_mesh_count") {
        anAttribute->CopyTo(fObject, &KGExtrudedObject::SetExtrudedMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "extruded_mesh_power") {
        anAttribute->CopyTo(fObject, &KGExtrudedObject::SetExtrudedMeshPower);
        return true;
    }
    if (anAttribute->GetName() == "refine_mesh") {
        anAttribute->CopyTo(fObject, &KGExtrudedObject::SetRefineMesh);
        return true;
    }
    if (anAttribute->GetName() == "mesh_merge_distance") {
        anAttribute->CopyTo(fObject, &KGExtrudedObject::SetMeshMergeDistance);
        return true;
    }
    if (anAttribute->GetName() == "closed_form") {
        bool closedLoops = true;
        anAttribute->CopyTo(closedLoops);
        if (closedLoops)
            fObject->Close();
        else
            fObject->Open();
        return true;
    }
    return false;
}

template<> inline bool KGExtrudedObjectBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "outer_line") {
        anElement->AsPointer<KGExtrudedObject::Line>()->Initialize();
        anElement->ReleaseTo(fObject, &KGExtrudedObject::AddOuterSegment);
        return true;
    }
    if (anElement->GetName() == "inner_line") {
        anElement->AsPointer<KGExtrudedObject::Line>()->Initialize();
        anElement->ReleaseTo(fObject, &KGExtrudedObject::AddInnerSegment);
        return true;
    }
    if (anElement->GetName() == "outer_arc") {
        anElement->AsPointer<KGExtrudedObject::Arc>()->Initialize();
        anElement->ReleaseTo(fObject, &KGExtrudedObject::AddOuterSegment);
        return true;
    }
    if (anElement->GetName() == "inner_arc") {
        anElement->AsPointer<KGExtrudedObject::Arc>()->Initialize();
        anElement->ReleaseTo(fObject, &KGExtrudedObject::AddInnerSegment);
        return true;
    }
    return false;
}

//! With this element it is possible to create an extruded surface (from a poly-loop like curve) with arbitrary holes in it.
//! The commands ``<inner_...>`` and ``<outer_...>`` define whether the extruded path will produce an outer or an inner
//! surface. Therefore, they define if there will be a hole in the surface and its shape.
//!
//! @image Documentation/gh-pages/source/_images/kgeobag_extruded_surface_model.png 400pt
//!
//! - Works with MeshViewer: Yes.
//! - Works with GeometryViewer: No.
//! @example
//! <extruded_surface name="example_extruded_surface">
//!     <extruded_object z_min="-1." z_max="1." longitudinal_mesh_count="40" longitudinal_mesh_power="2." closed_form="1">
//!     <inner_line x1="-.5" y1="-.5" x2="-.5" y2=".5"/>
//!         <inner_arc x1="-.5" y1=".5" x2=".5" y2=".5" radius="1." positive_orientation="0"/>
//!         <inner_arc x1=".5" y1=".5" x2=".5" y2="-.5" radius="1." positive_orientation="1"/>
//!         <inner_arc x1=".5" y1="-.5" x2="-.5" y2="-.5" radius="1." positive_orientation="0"/>
//!         <outer_arc x1="-1." y1="-1." x2="-1." y2="1." radius="{TMath::Sqrt(2.)}" positive_orientation="0"/>
//!         <outer_line x1="-1." y1="1." x2="1." y2="1."/>
//!         <outer_line x1="1." y1="1." x2="1." y2="-1."/>
//!         <outer_line x1="1." y1="-1." x2="-1." y2="-1."/>
//!     </extruded_object>
//! </extruded_surface>
using KGExtrudedSurfaceBuilder = KComplexElement<KGWrappedSurface<KGExtrudedObject>>;

template<> inline bool KGExtrudedSurfaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGWrappedSurface<KGExtrudedObject>::SetName);
        return true;
    }
    return false;
}

template<> inline bool KGExtrudedSurfaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "extruded_object") {
        KGExtrudedObject* object = nullptr;
        anElement->ReleaseTo(object);
        object->Initialize();
        std::shared_ptr<KGExtrudedObject> smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
}


//! For visualization, see :ref:`<extruded_surface> <binding-kgwrappedsurface-kgextrudedobject>`.
//!
//! - Works with GeometryViewer: No.
//! - Works with MeshViewer: No.
//! @example
//! <extruded_space name="example_extruded_space">
//!     <extruded_object z_min="-1." z_max="1." longitudinal_mesh_count="40" longitudinal_mesh_power="2." closed_form="1">
//!         <inner_line x1="-.5" y1="-.5" x2="-.5" y2=".5"/>
//!         <inner_arc x1="-.5" y1=".5" x2=".5" y2=".5" radius="1." positive_orientation="0"/>
//!         <inner_arc x1=".5" y1=".5" x2=".5" y2="-.5" radius="1." positive_orientation="1"/>
//!         <inner_arc x1=".5" y1="-.5" x2="-.5" y2="-.5" radius="1." positive_orientation="0"/>
//!         <outer_arc x1="-1." y1="-1." x2="-1." y2="1." radius="{TMath::Sqrt(2.)}" positive_orientation="0"/>
//!         <outer_line x1="-1." y1="1." x2="1." y2="1."/>
//!         <outer_line x1="1." y1="1." x2="1." y2="-1."/>
//!         <outer_line x1="1." y1="-1." x2="-1." y2="-1."/>
//!     </extruded_object>
//! </extruded_space>
using KGExtrudedSpaceBuilder = KComplexElement<KGWrappedSpace<KGExtrudedObject>>;

template<> inline bool KGExtrudedSpaceBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "name") {
        anAttribute->CopyTo(fObject, &KGWrappedSpace<KGExtrudedObject>::SetName);
        return true;
    }
    return false;
}

template<> inline bool KGExtrudedSpaceBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "extruded_object") {
        KGExtrudedObject* object = nullptr;
        anElement->ReleaseTo(object);
        object->Initialize();
        std::shared_ptr<KGExtrudedObject> smartPtr(object);
        fObject->SetObject(smartPtr);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
