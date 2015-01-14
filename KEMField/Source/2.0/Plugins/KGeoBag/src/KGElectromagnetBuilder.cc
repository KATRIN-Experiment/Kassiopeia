#include "KGElectromagnetBuilder.hh"

namespace KGeoBag
{

    KGElectromagnetAttributor::KGElectromagnetAttributor() :
            fSurfaces(),
            fSpaces(),
            fLineCurrent( 0.0 ),
            fScalingFactor( 1.0 ),
            fDirection( 1.0 )
    {
    }

    KGElectromagnetAttributor::~KGElectromagnetAttributor()
    {
        KGElectromagnetSurface* tElectromagnetSurface;
        for( vector< KGSurface* >::iterator tIt = fSurfaces.begin(); tIt != fSurfaces.end(); tIt++ )
        {
            tElectromagnetSurface = (*tIt)->MakeExtension< KGElectromagnet >();
            tElectromagnetSurface->SetName( GetName() );
            tElectromagnetSurface->SetTags( GetTags() );
            tElectromagnetSurface->SetCurrent( GetCurrent() );
        }
        KGElectromagnetSpace* tElectromagnetSpace;
        for( vector< KGSpace* >::iterator tIt = fSpaces.begin(); tIt != fSpaces.end(); tIt++ )
        {
            tElectromagnetSpace = (*tIt)->MakeExtension< KGElectromagnet >();
            tElectromagnetSpace->SetName( GetName() );
            tElectromagnetSpace->SetTags( GetTags() );
            tElectromagnetSpace->SetCurrent( GetCurrent() );
        }
    }

    void KGElectromagnetAttributor::AddSurface( KGSurface* aSurface )
    {
        fSurfaces.push_back( aSurface );
        return;
    }
    void KGElectromagnetAttributor::AddSpace( KGSpace* aSpace )
    {
        fSpaces.push_back( aSpace );
        return;
    }

}

#include "KGInterfaceBuilder.hh"

using namespace KGeoBag;
namespace katrin
{

    template< >
    inline KGElectromagnetBuilder::~KComplexElement()
    {
    }

    static int sKGElectromagnetStructure =
        KGElectromagnetBuilder::Attribute< string >( "name" ) +
        KGElectromagnetBuilder::Attribute< double >( "current" ) +
        KGElectromagnetBuilder::Attribute< double >( "scaling_factor" ) +
        KGElectromagnetBuilder::Attribute< string >( "direction" ) +
        KGElectromagnetBuilder::Attribute< string >( "surfaces" ) +
        KGElectromagnetBuilder::Attribute< string >( "spaces" );


    static int sKGElectromagnet =
        KGInterfaceBuilder::ComplexElement< KGElectromagnetAttributor >( "electromagnet" );

}
