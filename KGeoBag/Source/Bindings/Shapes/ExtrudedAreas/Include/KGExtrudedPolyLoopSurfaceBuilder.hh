#ifndef KGEXTRUDEDPOLYLOOPSURFACEBUILDER_HH_
#define KGEXTRUDEDPOLYLOOPSURFACEBUILDER_HH_

#include "KGPlanarPolyLoopBuilder.hh"
#include "KGExtrudedPolyLoopSurface.hh"

namespace katrin
{

    typedef KComplexElement< KGExtrudedPolyLoopSurface > KGExtrudedPolyLoopSurfaceBuilder;

    template< >
    inline bool KGExtrudedPolyLoopSurfaceBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == string( "name" ) )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedPolyLoopSurface::SetName );
            return true;
        }
        if( anAttribute->GetName() == string( "zmin" ) )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedPolyLoopSurface::ZMin );
            return true;
        }
        if( anAttribute->GetName() == string( "zmax" ) )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedPolyLoopSurface::ZMax );
            return true;
        }
        if( anAttribute->GetName() == string( "extruded_mesh_count" ) )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedPolyLoopSurface::ExtrudedMeshCount );
            return true;
        }
        if( anAttribute->GetName() == string( "extruded_mesh_power" ) )
        {
            anAttribute->CopyTo( fObject, &KGExtrudedPolyLoopSurface::ExtrudedMeshPower );
            return true;
        }
        return false;
    }

    template< >
    inline bool KGExtrudedPolyLoopSurfaceBuilder::AddElement( KContainer* anElement )
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
