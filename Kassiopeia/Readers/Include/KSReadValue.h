#ifndef Kassiopeia_KSReadValue_h_
#define Kassiopeia_KSReadValue_h_

#include "KSReadersMessage.h"

#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

#include "KTwoVector.hh"
using KGeoBag::KTwoVector;

namespace Kassiopeia
{

    template< class XType >
    class KSReadValue
    {
        public:
            typedef XType Type;

        public:
            static const KSReadValue< XType > sZero;

        public:
            KSReadValue();
            KSReadValue( const XType& aValue );
            KSReadValue( const KSReadValue< XType >& aValue );
            ~KSReadValue();

        public:
            XType& Value();
            const XType& Value() const;
            XType* Pointer();
            XType** Handle();

        private:
            XType fValue;
            XType* fPointer;
            XType** fHandle;

            //*********
            //operators
            //*********

        public:
            KSReadValue< XType >& operator=( const XType & aValue );
            KSReadValue< XType >& operator=( const KSReadValue< XType >& aValue );

            bool operator==( const KSReadValue< XType >& aRightValue );
            bool operator!=( const KSReadValue< XType >& aRightValue );
            bool operator<( const KSReadValue< XType >& aRightValue );
            bool operator<=( const KSReadValue< XType >& aRightValue );
            bool operator>( const KSReadValue< XType >& aRightValue );
            bool operator>=( const KSReadValue< XType >& aRightValue );

            KSReadValue< XType > operator+( const KSReadValue< XType >& aRightValue ) const;
            KSReadValue< XType > operator-( const KSReadValue< XType >& aRightValue ) const;
            KSReadValue< XType > operator*( const KSReadValue< XType >& aRightValue ) const;
            KSReadValue< XType > operator/( const KSReadValue< XType >& aRightValue ) const;

            KSReadValue< XType > operator+( const XType& aRightValue ) const;
            KSReadValue< XType > operator-( const XType& aRightValue ) const;
            KSReadValue< XType > operator*( const XType& aRightValue ) const;
            KSReadValue< XType > operator/( const XType& aRightValue ) const;
    };

    template< class XType >
    KSReadValue< XType >::KSReadValue() :
            fValue( sZero.fValue ),
            fPointer( &fValue ),
            fHandle( &fPointer )
    {
    }
    template< class XType >
    KSReadValue< XType >::KSReadValue( const XType& aValue ) :
            fValue( aValue ),
            fPointer( &fValue ),
            fHandle( &fPointer )
    {
    }
    template< class XType >
    KSReadValue< XType >::KSReadValue( const KSReadValue< XType >& aValue ) :
            fValue( aValue.fValue ),
            fPointer( &fValue ),
            fHandle( &fPointer )
    {
    }
    template< class XType >
    KSReadValue< XType >::~KSReadValue()
    {
    }

    template< class XType >
    XType& KSReadValue< XType >::Value()
    {
        return fValue;
    }
    template< class XType >
    const XType& KSReadValue< XType >::Value() const
    {
        return fValue;
    }
    template< class XType >
    XType* KSReadValue< XType >::Pointer()
    {
        return fPointer;
    }
    template< class XType >
    XType** KSReadValue< XType >::Handle()
    {
        return fHandle;
    }

    template< class XType >
    KSReadValue< XType >& KSReadValue< XType >::operator=( const KSReadValue< XType >& aValue )
    {
        fValue = aValue.fValue;
        return *this;
    }
    template< class XType >
    KSReadValue< XType >& KSReadValue< XType >::operator=( const XType& aValue )
    {
        fValue = aValue;
        return *this;
    }

    template< class XType >
    bool KSReadValue< XType >::operator==( const KSReadValue< XType >& aValue )
    {
        return (fValue == aValue.fValue);
    }
    template< class XType >
    bool KSReadValue< XType >::operator!=( const KSReadValue< XType >& aValue )
    {
        return (fValue != aValue.fValue);
    }
    template< class XType >
    bool KSReadValue< XType >::operator<( const KSReadValue< XType >& aValue )
    {
        return (fValue < aValue.fValue);
    }
    template< class XType >
    bool KSReadValue< XType >::operator<=( const KSReadValue< XType >& aValue )
    {
        return (fValue <= aValue.fValue);
    }
    template< class XType >
    bool KSReadValue< XType >::operator>( const KSReadValue< XType >& aValue )
    {
        return (fValue > aValue.fValue);
    }
    template< class XType >
    bool KSReadValue< XType >::operator>=( const KSReadValue< XType >& aValue )
    {
        return (fValue >= aValue.fValue);
    }

    template< class XType >
    KSReadValue< XType > KSReadValue< XType >::operator+( const KSReadValue< XType >& aValue ) const
    {
        return KSReadValue< XType >( fValue + aValue.fValue );
    }
    template< class XType >
    KSReadValue< XType > KSReadValue< XType >::operator-( const KSReadValue< XType >& aValue ) const
    {
        return KSReadValue< XType >( fValue - aValue.fValue );
    }
    template< class XType >
    KSReadValue< XType > KSReadValue< XType >::operator*( const KSReadValue< XType >& aValue ) const
    {
        return KSReadValue< XType >( fValue * aValue.fValue );
    }
    template< class XType >
    KSReadValue< XType > KSReadValue< XType >::operator/( const KSReadValue< XType >& aValue ) const
    {
        return KSReadValue< XType >( fValue / aValue.fValue );
    }

    template< class XType >
    KSReadValue< XType > KSReadValue< XType >::operator+( const XType& aValue ) const
    {
        return KSReadValue< XType >( fValue + aValue );
    }
    template< class XType >
    KSReadValue< XType > KSReadValue< XType >::operator-( const XType& aValue ) const
    {
        return KSReadValue< XType >( fValue - aValue );
    }
    template< class XType >
    KSReadValue< XType > KSReadValue< XType >::operator*( const XType& aValue ) const
    {
        return KSReadValue< XType >( fValue * aValue );
    }
    template< class XType >
    KSReadValue< XType > KSReadValue< XType >::operator/( const XType& aValue ) const
    {
        return KSReadValue< XType >( fValue / aValue );
    }

    typedef KSReadValue< bool > KSBool;
    typedef KSReadValue< unsigned char > KSUChar;
    typedef KSReadValue< char > KSChar;
    typedef KSReadValue< unsigned short > KSUShort;
    typedef KSReadValue< short > KSShort;
    typedef KSReadValue< unsigned int > KSUInt;
    typedef KSReadValue< int > KSInt;
    typedef KSReadValue< unsigned long > KSULong;
    typedef KSReadValue< long > KSLong;
    typedef KSReadValue< float > KSFloat;
    typedef KSReadValue< double > KSDouble;
    typedef KSReadValue< KThreeVector > KSThreeVector;
    typedef KSReadValue< KTwoVector > KSTwoVector;
    typedef KSReadValue< std::string > KSString;

    template< >
    const KSBool KSBool::sZero;

    template< >
    const KSUChar KSUChar::sZero;

    template< >
    const KSChar KSChar::sZero;

    template< >
    const KSUShort KSUShort::sZero;

    template< >
    const KSShort KSShort::sZero;

    template< >
    const KSUInt KSUInt::sZero;

    template< >
    const KSInt KSInt::sZero;

    template< >
    const KSULong KSULong::sZero;

    template< >
    const KSLong KSLong::sZero;

    template< >
    const KSFloat KSFloat::sZero;

    template< >
    const KSDouble KSDouble::sZero;

    template< >
    const KSThreeVector KSThreeVector::sZero;

    template< >
    const KSTwoVector KSTwoVector::sZero;

    template< >
    const KSString KSString::sZero;
}

#endif
