#include "KGAppearanceBuilder.hh"

namespace KGeoBag
{

    KGAppearanceAttributor::KGAppearanceAttributor() :
            fSurfaces(),
            fSpaces()
    {
    }

    KGAppearanceAttributor::~KGAppearanceAttributor()
    {
        KGAppearanceSurface* tAppearanceSurface;
        for( vector< KGSurface* >::iterator tIt = fSurfaces.begin(); tIt != fSurfaces.end(); tIt++ )
        {
            tAppearanceSurface = (*tIt)->MakeExtension< KGAppearance >();
            tAppearanceSurface->SetName( GetName() );
            tAppearanceSurface->SetTags( GetTags() );
            tAppearanceSurface->SetColor( GetColor() );
            tAppearanceSurface->SetArc( GetArc() );
        }
        KGAppearanceSpace* tAppearanceSpace;
        for( vector< KGSpace* >::iterator tIt = fSpaces.begin(); tIt != fSpaces.end(); tIt++ )
        {
            tAppearanceSpace = (*tIt)->MakeExtension< KGAppearance >();
            tAppearanceSpace->SetName( GetName() );
            tAppearanceSpace->SetTags( GetTags() );
            tAppearanceSpace->SetColor( GetColor() );
            tAppearanceSpace->SetArc( GetArc() );
        }
    }

    void KGAppearanceAttributor::AddSurface( KGSurface* aSurface )
    {
        fSurfaces.push_back( aSurface );
        return;
    }
    void KGAppearanceAttributor::AddSpace( KGSpace* aSpace )
    {
        fSpaces.push_back( aSpace );
        return;
    }

}

#include "KGInterfaceBuilder.hh"

namespace katrin
{

    template< >
    KGAppearanceBuilder::~KComplexElement()
    {
    }

    static const int sKGAppearanceStructure =
        KGAppearanceBuilder::Attribute< string >( "name" ) +
        KGAppearanceBuilder::Attribute< KGRGBAColor >( "color" ) +
        KGAppearanceBuilder::Attribute< unsigned int >( "arc" ) +
        KGAppearanceBuilder::Attribute< string >( "surfaces" ) +
        KGAppearanceBuilder::Attribute< string >( "spaces" );

    static const int sKGAppearance =
        KGInterfaceBuilder::ComplexElement< KGAppearanceAttributor >( "appearance" );

}
