#include "KGMeshBuilder.hh"
#include "KGMesher.hh"

namespace KGeoBag
{

    KGMeshAttributor::KGMeshAttributor() :
            fSurfaces(),
            fSpaces()
    {
    }

    KGMeshAttributor::~KGMeshAttributor()
    {
        KGMesher tMesher;

        KGMeshSurface* tMeshSurface;
        for( vector< KGSurface* >::iterator tIt = fSurfaces.begin(); tIt != fSurfaces.end(); tIt++ )
        {
            tMeshSurface = (*tIt)->MakeExtension< KGMesh >();
            (*tIt)->AcceptNode( &tMesher );
            tMeshSurface->SetName( GetName() );
            tMeshSurface->SetTags( GetTags() );
        }
        KGMeshSpace* tMeshSpace;
        for( vector< KGSpace* >::iterator tIt = fSpaces.begin(); tIt != fSpaces.end(); tIt++ )
        {
            tMeshSpace = (*tIt)->MakeExtension< KGMesh >();
            (*tIt)->AcceptNode( &tMesher );
            tMeshSpace->SetName( GetName() );
            tMeshSpace->SetTags( GetTags() );
        }
    }

    void KGMeshAttributor::AddSurface( KGSurface* aSurface )
    {
        fSurfaces.push_back( aSurface );
        return;
    }
    void KGMeshAttributor::AddSpace( KGSpace* aSpace )
    {
        fSpaces.push_back( aSpace );
        return;
    }

}

#include "KGInterfaceBuilder.hh"

namespace katrin
{

    template< >
    KGMeshBuilder::~KComplexElement()
    {
    }

    STATICINT sKGMeshStructure =
        KGMeshBuilder::Attribute< string >( "name" ) +
        KGMeshBuilder::Attribute< string >( "surfaces" ) +
        KGMeshBuilder::Attribute< string >( "spaces" );

    STATICINT sKGMesh =
      KGInterfaceBuilder::ComplexElement< KGMeshAttributor >( "mesh" );

}
