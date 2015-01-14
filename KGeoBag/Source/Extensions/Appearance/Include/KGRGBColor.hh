#ifndef KGRBGCOLOR_HH_
#define KGRBGCOLOR_HH_

#include <istream>
using std::istream;

#include <ostream>
using std::ostream;

namespace KGeoBag
{

    class KGRGBColor
    {
        public:
            KGRGBColor();
            KGRGBColor( const KGRGBColor& aColor );
            KGRGBColor( int aRed, int aGreen, int aBlue );
            virtual ~KGRGBColor();

            void SetRed( const unsigned char& aRed );
            const unsigned char& GetRed() const;

            void SetGreen( const unsigned char& aGreen );
            const unsigned char& GetGreen() const;

            void SetBlue( const unsigned char& aBlue );
            const unsigned char& GetBlue() const;

        private:
            unsigned char fRed;
            unsigned char fGreen;
            unsigned char fBlue;
    };

    inline istream& operator>>( istream& aStream, KGRGBColor& aColor )
    {
        unsigned int tRed;
        unsigned int tGreen;
        unsigned int tBlue;

        aStream >> tRed >> tGreen >> tBlue;

        aColor.SetRed( tRed );
        aColor.SetGreen( tGreen );
        aColor.SetBlue( tBlue );

        return aStream;
    }

    inline ostream& operator<<( ostream& aStream, KGRGBColor& aColor )
    {
        unsigned int tRed = aColor.GetRed();
        unsigned int tGreen = aColor.GetGreen();
        unsigned int tBlue = aColor.GetBlue();

        aStream << "<" << tRed << ", " << tGreen << ", " << tBlue << ">";

        return aStream;
    }

}

#endif
