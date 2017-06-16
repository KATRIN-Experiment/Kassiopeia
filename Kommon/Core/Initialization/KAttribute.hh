#ifndef Kommon_KAttribute_hh_
#define Kommon_KAttribute_hh_

#include "KAttributeBase.hh"

namespace katrin
{
    template< class XType >
    class KAttribute :
        public KAttributeBase
    {
        public:
            KAttribute( KElementBase* aParentElement = NULL );
            virtual ~KAttribute();

            virtual bool SetValue( KToken* aToken );

            static KAttributeBase* Create( KElementBase* aParentElement );

        private:
            XType* fObject;
    };

    template< class XType >
    KAttribute< XType >::KAttribute( KElementBase* aParentElement )
    {
        fParentElement = aParentElement;

        fObject = new XType();
        Set( fObject );
    }
    template< class XType >
    KAttribute< XType >::~KAttribute()
    {
    }

    template< class XType >
    bool KAttribute< XType >::SetValue( KToken* aToken )
    {
        (*fObject) = aToken->GetValue< XType >();
        return true;
    }

    template< class XType >
    KAttributeBase* KAttribute< XType >::Create( KElementBase* aParentElement )
    {
        return new KAttribute< XType >( aParentElement );
    }

}

#endif
