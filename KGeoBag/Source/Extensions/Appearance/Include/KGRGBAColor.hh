#ifndef KGRGBACOLOR_HH_
#define KGRGBACOLOR_HH_

#include "KGRGBColor.hh"

namespace KGeoBag
{

    class KGRGBAColor :
        public KGRGBColor
    {
        public:
            KGRGBAColor();
            KGRGBAColor( const KGRGBAColor& );
            KGRGBAColor( int aRed, int aGreen, int aBlue, int anOpacity );
            virtual ~KGRGBAColor();

            void SetOpacity( const unsigned char& anOpacity );
            const unsigned char& GetOpacity() const;

        private:
            unsigned char fOpacity;
    };

    inline istream& operator>>( istream& aStream, KGRGBAColor& aColor )
    {
        unsigned int tRed;
        unsigned int tGreen;
        unsigned int tBlue;
        unsigned int tOpacity;

        aStream >> tRed >> tGreen >> tBlue >> tOpacity;

        aColor.SetRed( tRed );
        aColor.SetGreen( tGreen );
        aColor.SetBlue( tBlue );
        aColor.SetOpacity( tOpacity );

        return aStream;
    }
    inline ostream& operator<<( ostream& aStream, const KGRGBAColor& aColor )
    {
        unsigned int tRed = aColor.GetRed();
        unsigned int tGreen = aColor.GetGreen();
        unsigned int tBlue = aColor.GetBlue();
        unsigned int tOpacity = aColor.GetOpacity();

        aStream << "<" << tRed << ", " << tGreen << ", " << tBlue << "; " << tOpacity << ">";

        return aStream;
    }
    inline bool operator!=( KGRGBAColor& aColor1, KGRGBAColor& aColor2 )
    {
        if ( aColor1.GetRed() != aColor2.GetRed() )
            return true;
        if ( aColor1.GetGreen() != aColor2.GetGreen() )
            return true;
        if ( aColor1.GetBlue() != aColor2.GetBlue() )
            return true;
        if ( aColor1.GetOpacity() != aColor2.GetOpacity() )
            return true;

        return false;
    }
}

#endif
