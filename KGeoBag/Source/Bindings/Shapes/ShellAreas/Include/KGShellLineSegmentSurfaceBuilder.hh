#ifndef KGSHELLLINESEGMENTSURFACEBUILDER_HH_
#define KGSHELLLINESEGMENTSURFACEBUILDER_HH_

#include "KGPlanarLineSegmentBuilder.hh"
#include "KGShellLineSegmentSurface.hh"

namespace katrin
{

    typedef KComplexElement< KGShellLineSegmentSurface > KGShellLineSegmentSurfaceBuilder;

    template< >
    inline bool KGShellLineSegmentSurfaceBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == "name" )
        {
            anAttribute->CopyTo( fObject, &KGShellLineSegmentSurface::SetName );
            return true;
        }
        if( anAttribute->GetName() == "angle_start" )
        {
            anAttribute->CopyTo( fObject, &KGShellLineSegmentSurface::AngleStart );
            return true;
        }
        if( anAttribute->GetName() == "angle_stop" )
        {
            anAttribute->CopyTo( fObject, &KGShellLineSegmentSurface::AngleStop );
            return true;
        }
        if( anAttribute->GetName() == "shell_mesh_count" )
        {
            anAttribute->CopyTo( fObject, &KGShellLineSegmentSurface::ShellMeshCount );
            return true;
        }
        if( anAttribute->GetName() == "shell_mesh_power" )
        {
            anAttribute->CopyTo( fObject, &KGShellLineSegmentSurface::ShellMeshPower );
            return true;
        }
        return false;
    }

    template< >
    inline bool KGShellLineSegmentSurfaceBuilder::AddElement( KContainer* anElement )
    {
        if( anElement->GetName() == "line_segment" )
        {
            anElement->CopyTo( fObject->Path().operator ->(), &KGPlanarLineSegment::CopyFrom );
            return true;
        }
        return false;
    }

}

#endif
