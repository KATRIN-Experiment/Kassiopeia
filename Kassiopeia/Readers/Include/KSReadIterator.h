#ifndef Kassiopeia_KSReadIterator_h_
#define Kassiopeia_KSReadIterator_h_

#include "KSReadSet.h"

namespace Kassiopeia
{

    class KSReadIterator
    {
        public:
            KSReadIterator();
            virtual ~KSReadIterator();

        public:
            virtual void operator++( int ) = 0;
            virtual void operator--( int ) = 0;
            virtual void operator<<( const unsigned int& aValue ) = 0;

        public:
            virtual bool Valid() const = 0;
            virtual unsigned int Index() const = 0;
            virtual bool operator<( const unsigned int& aValue ) const = 0;
            virtual bool operator<=( const unsigned int& aValue ) const = 0;
            virtual bool operator>( const unsigned int& aValue ) const = 0;
            virtual bool operator>=( const unsigned int& aValue ) const = 0;
            virtual bool operator==( const unsigned int& aValue ) const = 0;
            virtual bool operator!=( const unsigned int& aValue ) const = 0;

        public:
            template< class XType >
            XType& Add( const string& aVariable );

            template< class XType >
            XType& Get( const string& aVariable ) const;
    };

    template< class XType >
    XType& KSReadIterator::Add( const string& aVariable )
    {
        KSReadSet< XType >& tSet = dynamic_cast< KSReadSet< XType >& >( *this );
        return tSet.Add( aVariable );
    }

    template< class XType >
    XType& KSReadIterator::Get( const string& aVariable ) const
    {
        const KSReadSet< XType >& tSet = dynamic_cast< const KSReadSet< XType >& >( *this );
        return tSet.Get( aVariable );
    }

}

#endif
