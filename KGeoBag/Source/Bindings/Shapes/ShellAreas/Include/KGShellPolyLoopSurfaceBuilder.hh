#ifndef KGSHELLPOLYLOOPSURFACEBUILDER_HH_
#define KGSHELLPOLYLOOPSURFACEBUILDER_HH_

#include "KGPlanarPolyLoopBuilder.hh"
#include "KGShellPolyLoopSurface.hh"

namespace katrin
{

    typedef KComplexElement< KGShellPolyLoopSurface > KGShellPolyLoopSurfaceBuilder;

    template< >
    inline bool KGShellPolyLoopSurfaceBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == string( "name" ) )
        {
            anAttribute->CopyTo( fObject, &KGShellPolyLoopSurface::SetName );
            return true;
        }
        if( anAttribute->GetName() == string( "angle_start" ) )
        {
            anAttribute->CopyTo( fObject, &KGShellPolyLoopSurface::AngleStart );
            return true;
        }
        if( anAttribute->GetName() == string( "angle_stop" ) )
        {
            anAttribute->CopyTo( fObject, &KGShellPolyLoopSurface::AngleStop );
            return true;
        }
        if( anAttribute->GetName() == string( "shell_mesh_count" ) )
        {
            anAttribute->CopyTo( fObject, &KGShellPolyLoopSurface::ShellMeshCount );
            return true;
        }
         if( anAttribute->GetName() == string( "shell_mesh_power" ) )
        {
            anAttribute->CopyTo( fObject, &KGShellPolyLoopSurface::ShellMeshPower );
            return true;
        }
        return false;
    }

    template< >
    inline bool KGShellPolyLoopSurfaceBuilder::AddElement( KContainer* anElement )
    {
        if( anElement->GetName() == string( "poly_loop" ) )
        {
            anElement->CopyTo( fObject->Path().operator ->(), &KGPlanarPolyLoop::CopyFrom );
            return true;
        }
        return false;
    }

}

#endif
