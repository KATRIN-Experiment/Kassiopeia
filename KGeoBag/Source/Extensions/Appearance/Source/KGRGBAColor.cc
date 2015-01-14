#include "KGRGBAColor.hh"

namespace KGeoBag
{

    KGRGBAColor::KGRGBAColor() :
            KGRGBColor(),
            fOpacity( 255 )
    {
    }
    KGRGBAColor::KGRGBAColor( int aRed, int aGreen, int aBlue, int anOpacity ) :
            KGRGBColor( aRed, aGreen, aBlue ),
            fOpacity( anOpacity )
    {
    }
    KGRGBAColor::KGRGBAColor( const KGRGBAColor& aColor ) :
            KGRGBColor( aColor ),
            fOpacity( aColor.fOpacity )
    {
    }
    KGRGBAColor::~KGRGBAColor()
    {
    }

    void KGRGBAColor::SetOpacity( const unsigned char& anOpacity )
    {
        fOpacity = anOpacity;
        return;
    }
    const unsigned char& KGRGBAColor::GetOpacity() const
    {
        return fOpacity;
    }

}
