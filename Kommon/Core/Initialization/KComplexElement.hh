#ifndef Kommon_KComplexElement_hh_
#define Kommon_KComplexElement_hh_

#include "KElementBase.hh"
#include "KAttributeBase.hh"

#include "KAttribute.hh"
#include "KSimpleElement.hh"

namespace katrin
{

    template< class XType >
    class KComplexElement :
        public KElementBase
    {
        public:
            KComplexElement( KElementBase* aParentElement = NULL );
            virtual ~KComplexElement();

            virtual bool Begin();
            virtual bool AddAttribute( KContainer* aToken );
            virtual bool Body();
            virtual bool AddElement( KContainer* anElement );
            virtual bool SetValue( KToken* aValue );
            virtual bool End();

            static KElementBase* Create( KElementBase* aParentElement );
            template< class XAttributeType >
            static int Attribute( const string& aName );
            template< class XElementType >
            static int SimpleElement( const string& aName );
            template< class XElementType >
            static int ComplexElement( const string& aName );

        protected:
            XType* fObject;
            static KAttributeMap* sAttributes;
            static KElementMap* sElements;
    };

    template< class XType >
    KComplexElement< XType >::KComplexElement( KElementBase* aParentElement ) :
        fObject( NULL )
    {
        fParentElement = aParentElement;

        if( sElements == NULL )
        {
            sElements = new KElementMap();
        }
        fElements = sElements;

        if( sAttributes == NULL )
        {
            sAttributes = new KAttributeMap();
        }
        fAttributes = sAttributes;
    }
    template< class XType >
    KComplexElement< XType >::~KComplexElement()
    {
    }

    template< class XType >
    bool KComplexElement< XType >::Begin()
    {
        fObject = new XType();
        Set( fObject );
        return true;
    }
    template< class XType >
    bool KComplexElement< XType >::AddAttribute( KContainer* )
    {
        return true;
    }
    template< class XType >
    bool KComplexElement< XType >::Body()
    {
        return true;
    }
    template< class XType >
    bool KComplexElement< XType >::AddElement( KContainer* )
    {
        return true;
    }
    template< class XType >
    bool KComplexElement< XType >::SetValue( KToken* )
    {
        return true;
    }
    template< class XType >
    bool KComplexElement< XType >::End()
    {
        return true;
    }

    template< class XType >
    KElementBase* KComplexElement< XType >::Create( KElementBase* aParentElement )
    {
        return new KComplexElement< XType >( aParentElement );
    }

    template< class XType >
    KAttributeMap* KComplexElement< XType >::sAttributes = NULL;
    template< class XType >
    template< class XAttributeType >
    int KComplexElement< XType >::Attribute( const string& aName )
    {
        if( sAttributes == NULL )
        {
            sAttributes = new KAttributeMap();
        }
        KComplexElement< XType >::sAttributes->insert( KAttributeEntry( aName, &KAttribute< XAttributeType >::Create ) );
        return 0;
    }
    template< class XType >
    KElementMap* KComplexElement< XType >::sElements = NULL;
    template< class XType >
    template< class XElementType >
    int KComplexElement< XType >::SimpleElement( const string& aName )
    {
        if( sElements == NULL )
        {
            sElements = new KElementMap();
        }
        KComplexElement< XType >::sElements->insert( KElementEntry( aName, &KSimpleElement< XElementType >::Create ) );
        return 0;
    }
    template< class XType >
    template< class XElementType >
    int KComplexElement< XType >::ComplexElement( const string& aName )
    {
        if( sElements == NULL )
        {
            sElements = new KElementMap();
        }
        KComplexElement< XType >::sElements->insert( KElementEntry( aName, &KComplexElement< XElementType >::Create ) );
        return 0;
    }
}

#endif
