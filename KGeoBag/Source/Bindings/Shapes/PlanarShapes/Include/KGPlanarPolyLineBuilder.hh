#ifndef KGPLANARPOLYLINEBUILDER_HH_
#define KGPLANARPOLYLINEBUILDER_HH_

#include "KComplexElement.hh"

#include "KGPlanarPolyLine.hh"
using namespace KGeoBag;

namespace katrin
{

    typedef KComplexElement< KGPlanarPolyLine::StartPointArguments > KGPlanarPolyLineStartPointArgumentsBuilder;

    template< >
    inline bool KGPlanarPolyLineStartPointArgumentsBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == "x" )
        {
            anAttribute->CopyTo( fObject->fPoint.X() );
            return true;
        }
        if( anAttribute->GetName() == "y" )
        {
            anAttribute->CopyTo( fObject->fPoint.Y() );
            return true;
        }
        return false;
    }

    typedef KComplexElement< KGPlanarPolyLine::LineArguments > KGPlanarPolyLineLineArgumentsBuilder;

    template< >
    inline bool KGPlanarPolyLineLineArgumentsBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == "x" )
        {
            anAttribute->CopyTo( fObject->fVertex.X() );
            return true;
        }
        if( anAttribute->GetName() == "y" )
        {
            anAttribute->CopyTo( fObject->fVertex.Y() );
            return true;
        }
        if( anAttribute->GetName() == "line_mesh_count" )
        {
            anAttribute->CopyTo( fObject->fMeshCount );
            return true;
        }
        if( anAttribute->GetName() == "line_mesh_power" )
        {
            anAttribute->CopyTo( fObject->fMeshPower );
            return true;
        }
        return false;
    }

    typedef KComplexElement< KGPlanarPolyLine::ArcArguments > KGPlanarPolyLineArcArgumentsBuilder;

    template< >
    inline bool KGPlanarPolyLineArcArgumentsBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == "x" )
        {
            anAttribute->CopyTo( fObject->fVertex.X() );
            return true;
        }
        if( anAttribute->GetName() == "y" )
        {
            anAttribute->CopyTo( fObject->fVertex.Y() );
            return true;
        }
        if( anAttribute->GetName() == "radius" )
        {
            anAttribute->CopyTo( fObject->fRadius );
            return true;
        }
        if( anAttribute->GetName() == "right" )
        {
            anAttribute->CopyTo( fObject->fRight );
            return true;
        }
        if( anAttribute->GetName() == "short" )
        {
            anAttribute->CopyTo( fObject->fShort );
            return true;
        }
        if( anAttribute->GetName() == "arc_mesh_count" )
        {
            anAttribute->CopyTo( fObject->fMeshCount );
            return true;
        }
        return false;
    }

    typedef KComplexElement< KGPlanarPolyLine > KGPlanarPolyLineBuilder;

    template< >
    inline bool KGPlanarPolyLineBuilder::AddAttribute( KContainer* /*anAttribute*/ )
    {
        return false;
    }

    template< >
    inline bool KGPlanarPolyLineBuilder::AddElement( KContainer* anElement )
    {
        if( anElement->GetName() == "start_point" )
        {
            KGPlanarPolyLine::StartPointArguments* tArgs = anElement->AsPointer< KGPlanarPolyLine::StartPointArguments >();
            fObject->StartPoint( tArgs->fPoint );
            return true;
        }
        if( anElement->GetName() == "next_line" )
        {
            KGPlanarPolyLine::LineArguments* tArgs = anElement->AsPointer< KGPlanarPolyLine::LineArguments >();
            fObject->NextLine( tArgs->fVertex, tArgs->fMeshCount, tArgs->fMeshPower );
            return true;
        }
        if( anElement->GetName() == "next_arc" )
        {
            KGPlanarPolyLine::ArcArguments* tArgs = anElement->AsPointer< KGPlanarPolyLine::ArcArguments >();
            fObject->NextArc( tArgs->fVertex, tArgs->fRadius, tArgs->fRight, tArgs->fShort, tArgs->fMeshCount );
            return true;
        }
        if( anElement->GetName() == "previous_line" )
        {
            KGPlanarPolyLine::LineArguments* tArgs = anElement->AsPointer< KGPlanarPolyLine::LineArguments >();
            fObject->PreviousLine( tArgs->fVertex, tArgs->fMeshCount, tArgs->fMeshPower );
            return true;
        }
        if( anElement->GetName() == "previous_arc" )
        {
            KGPlanarPolyLine::ArcArguments* tArgs = anElement->AsPointer< KGPlanarPolyLine::ArcArguments >();
            fObject->PreviousArc( tArgs->fVertex, tArgs->fRadius, tArgs->fRight, tArgs->fShort, tArgs->fMeshCount );
            return true;
        }
        return false;
    }

}

#endif
