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
            XType& Add( const std::string& aVariable );

            template< class XType >
            XType& Get( const std::string& aVariable ) const;

            template< class XType >
            bool Exists( const std::string& aVariable ) const;
    };

    template< class XType >
    XType& KSReadIterator::Add( const std::string& aVariable )
    {
        KSReadSet< XType >& tSet = dynamic_cast< KSReadSet< XType >& >( *this );
        return tSet.Add( aVariable );
    }

    template< class XType >
    XType& KSReadIterator::Get( const std::string& aVariable ) const
    {
        const KSReadSet< XType >& tSet = dynamic_cast< const KSReadSet< XType >& >( *this );
        return tSet.Get( aVariable );
    }

    template< class XType >
    bool KSReadIterator::Exists( const std::string& aVariable ) const
    {
        const KSReadSet< XType >& tSet = dynamic_cast< const KSReadSet< XType >& >( *this );
        return tSet.Exists( aVariable );
    }

}

#endif
