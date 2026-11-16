#ifndef KGPLANARPOLYLINEBUILDER_HH_
#define KGPLANARPOLYLINEBUILDER_HH_

#include "KComplexElement.hh"
#include "KGPlanarPolyLine.hh"
using namespace KGeoBag;

namespace katrin
{

typedef KComplexElement<KGPlanarPolyLine::StartPointArguments> KGPlanarPolyLineStartPointArgumentsBuilder;

template<> inline bool KGPlanarPolyLineStartPointArgumentsBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "x") {
        anAttribute->CopyTo(fObject->fPoint.X());
        return true;
    }
    if (anAttribute->GetName() == "y") {
        anAttribute->CopyTo(fObject->fPoint.Y());
        return true;
    }
    return false;
}

using KGPlanarPolyLineLineArgumentsBuilder = KComplexElement<KGPlanarPolyLine::LineArguments>;

template<> inline bool KGPlanarPolyLineLineArgumentsBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "x") {
        anAttribute->CopyTo(fObject->fVertex.X());
        return true;
    }
    if (anAttribute->GetName() == "y") {
        anAttribute->CopyTo(fObject->fVertex.Y());
        return true;
    }
    if (anAttribute->GetName() == "line_mesh_count") {
        anAttribute->CopyTo(fObject->fMeshCount);
        return true;
    }
    if (anAttribute->GetName() == "line_mesh_power") {
        anAttribute->CopyTo(fObject->fMeshPower);
        return true;
    }
    return false;
}

using KGPlanarPolyLineArcArgumentsBuilder = KComplexElement<KGPlanarPolyLine::ArcArguments>;

template<> inline bool KGPlanarPolyLineArcArgumentsBuilder::AddAttribute(KContainer* anAttribute)
{
    if (anAttribute->GetName() == "x") {
        anAttribute->CopyTo(fObject->fVertex.X());
        return true;
    }
    if (anAttribute->GetName() == "y") {
        anAttribute->CopyTo(fObject->fVertex.Y());
        return true;
    }
    if (anAttribute->GetName() == "radius") {
        anAttribute->CopyTo(fObject->fRadius);
        return true;
    }
    if (anAttribute->GetName() == "right") {
        anAttribute->CopyTo(fObject->fRight);
        return true;
    }
    if (anAttribute->GetName() == "short") {
        anAttribute->CopyTo(fObject->fShort);
        return true;
    }
    if (anAttribute->GetName() == "arc_mesh_count") {
        anAttribute->CopyTo(fObject->fMeshCount);
        return true;
    }
    return false;
}

//! This element is desribed using the same procedure as in poly-loop. However, the first and last points are not
//! necessarily connected. An XML example is below:
//!
//! .. code-block:: xml
//!
//!     <poly_line>
//!         <start_point x="0.2" y="0.3"/>
//!         <next_line x="0.1" y="0.1" line_mesh_count="36" line_mesh_power="4.5"/>
//!         <next_arc x="-0.1" y="0.1" radius="0.2" right="true" short="true" arc_mesh_count="96"/>
//!         <next_line x="-0.2" y="0.2" line_mesh_count="36" line_mesh_power="1."/>
//!         <next_line x="-0.3" y="0.1" line_mesh_count="24" line_mesh_power="4.5"/>
//!     </poly_line>
//!
//! In this case, the sequence progresses through five (x,y) points but does not connect back to the starting point. Hence,
//! it does not form a closed loop like the poly-loop element does, and does not need a ``<last_...>`` command. Otherwise,
//! the parameters for poly-line are the same as for poly-loop.
using KGPlanarPolyLineBuilder = KComplexElement<KGPlanarPolyLine>;

template<> inline bool KGPlanarPolyLineBuilder::AddAttribute(KContainer* /*anAttribute*/)
{
    return false;
}

template<> inline bool KGPlanarPolyLineBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "start_point") {
        auto* tArgs = anElement->AsPointer<KGPlanarPolyLine::StartPointArguments>();
        fObject->StartPoint(tArgs->fPoint);
        return true;
    }
    if (anElement->GetName() == "next_line") {
        auto* tArgs = anElement->AsPointer<KGPlanarPolyLine::LineArguments>();
        fObject->NextLine(tArgs->fVertex, tArgs->fMeshCount, tArgs->fMeshPower);
        return true;
    }
    if (anElement->GetName() == "next_arc") {
        auto* tArgs = anElement->AsPointer<KGPlanarPolyLine::ArcArguments>();
        fObject->NextArc(tArgs->fVertex, tArgs->fRadius, tArgs->fRight, tArgs->fShort, tArgs->fMeshCount);
        return true;
    }
    if (anElement->GetName() == "previous_line") {
        auto* tArgs = anElement->AsPointer<KGPlanarPolyLine::LineArguments>();
        fObject->PreviousLine(tArgs->fVertex, tArgs->fMeshCount, tArgs->fMeshPower);
        return true;
    }
    if (anElement->GetName() == "previous_arc") {
        auto* tArgs = anElement->AsPointer<KGPlanarPolyLine::ArcArguments>();
        fObject->PreviousArc(tArgs->fVertex, tArgs->fRadius, tArgs->fRight, tArgs->fShort, tArgs->fMeshCount);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
