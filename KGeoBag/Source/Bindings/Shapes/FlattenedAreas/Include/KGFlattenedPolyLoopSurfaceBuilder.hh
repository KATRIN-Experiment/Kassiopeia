#ifndef KGFLATTENEDPOLYLINESURFACEBUILDER_HH_
#define KGFLATTENEDPOLYLINESURFACEBUILDER_HH_

#include "KGPlanarPolyLoopBuilder.hh"
#include "KGFlattenedPolyLoopSurface.hh"

namespace katrin
{

    typedef KComplexElement< KGFlattenedPolyLoopSurface > KGFlattenedPolyLoopSurfaceBuilder;

    template< >
    inline bool KGFlattenedPolyLoopSurfaceBuilder::AddAttribute( KContainer* anAttribute )
    {
        if( anAttribute->GetName() == string( "name" ) )
        {
            anAttribute->CopyTo( fObject, &KGFlattenedPolyLoopSurface::SetName );
            return true;
        }
        if( anAttribute->GetName() == string( "z" ) )
        {
            anAttribute->CopyTo( fObject, &KGFlattenedPolyLoopSurface::Z );
            return true;
        }
        if( anAttribute->GetName() == string( "flattened_mesh_count" ) )
        {
            anAttribute->CopyTo( fObject, &KGFlattenedPolyLoopSurface::FlattenedMeshCount );
            return true;
        }
        if( anAttribute->GetName() == string( "flattened_mesh_power" ) )
        {
            anAttribute->CopyTo( fObject, &KGFlattenedPolyLoopSurface::FlattenedMeshPower );
            return true;
        }
        return false;
    }

    template< >
    inline bool KGFlattenedPolyLoopSurfaceBuilder::AddElement( KContainer* anElement )
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
