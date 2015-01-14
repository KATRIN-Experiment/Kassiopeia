#ifndef Kommon_KSimpleElement_hh_
#define Kommon_KSimpleElement_hh_

#include "KElementBase.hh"

namespace katrin
{

    template< class XType >
    class KSimpleElement :
        public KElementBase
    {
        public:
            KSimpleElement( KElementBase* aParentElement = NULL );
            virtual ~KSimpleElement();

            virtual bool Begin();
            virtual bool AddAttribute( KContainer* aToken );
            virtual bool Body();
            virtual bool AddElement( KContainer* anElement );
            virtual bool SetValue( KToken* aToken );
            virtual bool End();

            static KElementBase* Create( KElementBase* aParentElement );

        protected:
            XType* fObject;
            static KAttributeMap* sAttributes;
            static KElementMap* sElements;
    };

    template< class XType >
    KSimpleElement< XType >::KSimpleElement( KElementBase* aParentElement ) :
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
    KSimpleElement< XType >::~KSimpleElement()
    {
    }

    template< class XType >
    bool KSimpleElement< XType >::Begin()
    {
        fObject = new XType();
        Set( fObject );
        return true;
    }
    template< class XType >
    bool KSimpleElement< XType >::AddAttribute( KContainer* )
    {
        return false;
    }
    template< class XType >
    bool KSimpleElement< XType >::Body()
    {
        return true;
    }
    template< class XType >
    bool KSimpleElement< XType >::AddElement( KContainer* )
    {
        return false;
    }
    template< class XType >
    bool KSimpleElement< XType >::SetValue( KToken* aToken )
    {
        (*fObject) = aToken->GetValue< XType >();
        return true;
    }
    template< class XType >
    bool KSimpleElement< XType >::End()
    {
        return true;
    }

    template< class XType >
    KElementBase* KSimpleElement< XType >::Create( KElementBase* aParentElement )
    {
        return new KSimpleElement< XType >( aParentElement );
    }

    template< class XType >
    KAttributeMap* KSimpleElement< XType >::sAttributes = NULL;

    template< class XType >
    KElementMap* KSimpleElement< XType >::sElements = NULL;
}

#endif
