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

    inline ostream& operator<<( ostream& aStream, KGRGBAColor& aColor )
    {
        unsigned int tRed = aColor.GetRed();
        unsigned int tGreen = aColor.GetGreen();
        unsigned int tBlue = aColor.GetBlue();
        unsigned int tOpacity = aColor.GetOpacity();

        aStream << "<" << tRed << ", " << tGreen << ", " << tBlue << "; " << tOpacity << ">";

        return aStream;
    }

}

#endif
