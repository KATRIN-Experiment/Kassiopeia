#ifndef KGSHELLARCSEGMENTSURFACEBUILDER_HH_
#define KGSHELLARCSEGMENTSURFACEBUILDER_HH_

#include "KGPlanarArcSegmentBuilder.hh"
#include "KGShellArcSegmentSurface.hh"

namespace katrin
{

    typedef KComplexElement< KGShellArcSegmentSurface > KGShellArcSegmentSurfaceBuilder;

    template< >
    inline bool KGShellArcSegmentSurfaceBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == "name" )
        {
            anAttribute->CopyTo( fObject, &KGShellArcSegmentSurface::SetName );
            return true;
        }
        if( anAttribute->GetName() == "angle_start" )
        {
            anAttribute->CopyTo( fObject, &KGShellArcSegmentSurface::AngleStart );
            return true;
        }
        if( anAttribute->GetName() == "angle_stop" )
        {
            anAttribute->CopyTo( fObject, &KGShellArcSegmentSurface::AngleStop );
            return true;
        }
        if( anAttribute->GetName() == "shell_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGShellArcSegmentSurface::ShellMeshCount );
            return true;
        }
         if( anAttribute->GetName() == "shell_mesh_power" )
        {
            anAttribute->CopyTo( fObject, &KGShellArcSegmentSurface::ShellMeshPower );
            return true;
        }
        return false;
    }

    template< >
    inline bool KGShellArcSegmentSurfaceBuilder::AddElement( KContainer* anElement )
    {
        if( anElement->GetName() == "arc_segment" )
        {
            anElement->CopyTo( fObject->Path().operator ->(), &KGPlanarArcSegment::CopyFrom );
            return true;
        }
        return false;
    }

}

#endif
