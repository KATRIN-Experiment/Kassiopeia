#ifndef KGPLANARPOLYLOOPBUILDER_HH_
#define KGPLANARPOLYLOOPBUILDER_HH_

#include "KComplexElement.hh"
#include "KGPlanarPolyLoop.hh"
using namespace KGeoBag;

namespace katrin
{

typedef KComplexElement<KGPlanarPolyLoop::StartPointArguments> KGPlanarPolyLoopStartPointArgumentsBuilder;

template<> inline bool KGPlanarPolyLoopStartPointArgumentsBuilder::AddAttribute(KContainer* anAttribute)
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

using KGPlanarPolyLoopLineArgumentsBuilder = KComplexElement<KGPlanarPolyLoop::LineArguments>;

template<> inline bool KGPlanarPolyLoopLineArgumentsBuilder::AddAttribute(KContainer* anAttribute)
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


using KGPlanarPolyLoopArcArgumentsBuilder = KComplexElement<KGPlanarPolyLoop::ArcArguments>;

template<> inline bool KGPlanarPolyLoopArcArgumentsBuilder::AddAttribute(KContainer* anAttribute)
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


using KGPlanarPolyLoopLastLineArgumentsBuilder = KComplexElement<KGPlanarPolyLoop::LastLineArguments>;

template<> inline bool KGPlanarPolyLoopLastLineArgumentsBuilder::AddAttribute(KContainer* anAttribute)
{
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


using KGPlanarPolyLoopLastArcArgumentsBuilder = KComplexElement<KGPlanarPolyLoop::LastArcArguments>;

template<> inline bool KGPlanarPolyLoopLastArcArgumentsBuilder::AddAttribute(KContainer* anAttribute)
{
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


using KGPlanarPolyLoopBuilder = KComplexElement<KGPlanarPolyLoop>;

template<> inline bool KGPlanarPolyLoopBuilder::AddAttribute(KContainer* /*anAttribute*/)
{
    return false;
}

template<> inline bool KGPlanarPolyLoopBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "start_point") {
        auto* tArgs = anElement->AsPointer<KGPlanarPolyLoop::StartPointArguments>();
        fObject->StartPoint(tArgs->fPoint);
        return true;
    }
    if (anElement->GetName() == "next_line") {
        auto* tArgs = anElement->AsPointer<KGPlanarPolyLoop::LineArguments>();
        fObject->NextLine(tArgs->fVertex, tArgs->fMeshCount, tArgs->fMeshPower);
        return true;
    }
    if (anElement->GetName() == "next_arc") {
        auto* tArgs = anElement->AsPointer<KGPlanarPolyLoop::ArcArguments>();
        fObject->NextArc(tArgs->fVertex, tArgs->fRadius, tArgs->fRight, tArgs->fShort, tArgs->fMeshCount);
        return true;
    }
    if (anElement->GetName() == "previous_line") {
        auto* tArgs = anElement->AsPointer<KGPlanarPolyLoop::LineArguments>();
        fObject->PreviousLine(tArgs->fVertex, tArgs->fMeshCount, tArgs->fMeshPower);
        return true;
    }
    if (anElement->GetName() == "previous_arc") {
        auto* tArgs = anElement->AsPointer<KGPlanarPolyLoop::ArcArguments>();
        fObject->PreviousArc(tArgs->fVertex, tArgs->fRadius, tArgs->fRight, tArgs->fShort, tArgs->fMeshCount);
        return true;
    }
    if (anElement->GetName() == "last_line") {
        auto* tArgs = anElement->AsPointer<KGPlanarPolyLoop::LastLineArguments>();
        fObject->LastLine(tArgs->fMeshCount, tArgs->fMeshPower);
        return true;
    }
    if (anElement->GetName() == "last_arc") {
        auto* tArgs = anElement->AsPointer<KGPlanarPolyLoop::LastArcArguments>();
        fObject->LastArc(tArgs->fRadius, tArgs->fRight, tArgs->fShort, tArgs->fMeshCount);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
