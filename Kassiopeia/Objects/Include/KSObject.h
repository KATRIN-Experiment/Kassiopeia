#ifndef Kassiopeia_KSObject_h_
#define Kassiopeia_KSObject_h_

#include "KTagged.h"
using katrin::KTagged;

#include "KSObjectsMessage.h"

namespace Kassiopeia
{

    class KSObject :
        public KTagged
    {
        public:
            KSObject();
            KSObject( const KSObject& aCopy );
            virtual ~KSObject();

        public:
            virtual KSObject* Clone() const = 0;

        public:
            template< class XType >
            bool Is();

            template< class XType >
            bool Is() const;

            template< class XType >
            XType* As();

            template< class XType >
            const XType* As() const;

        protected:
            template< class XType >
            void Set( XType* );

        private:
            class KSHolder
            {
                public:
                    KSHolder();
                    virtual ~KSHolder();

                public:
                    virtual void Type() = 0;
            };

            template< class XType >
            class KSHolderTemplate :
                public KSHolder
            {
                public:
                    KSHolderTemplate( XType* anObject );
                    virtual ~KSHolderTemplate();

                public:
                    virtual void Type();

                private:
                    XType* fObject;
            };

            mutable KSHolder* fHolder;
    };

    inline KSObject::KSHolder::KSHolder()
    {
    }
    inline KSObject::KSHolder::~KSHolder()
    {
    }

    template< class XType >
    inline KSObject::KSHolderTemplate< XType >::KSHolderTemplate( XType* anObject ) :
            fObject( anObject )
    {
    }
    template< class XType >
    inline KSObject::KSHolderTemplate< XType >::~KSHolderTemplate()
    {
    }
    template< class XType >
    inline void KSObject::KSHolderTemplate< XType >::Type()
    {
        throw fObject;
        return;
    }

    template< class XType >
    inline bool KSObject::Is()
    {
        try
        {
            fHolder->Type();
        }
        catch( XType* tObject )
        {
            return true;
        }
        catch( ... )
        {
            return false;
        }
        return false;
    }
    template< >
    inline bool KSObject::Is< KSObject >()
    {
        return true;
    }

    template< class XType >
    inline bool KSObject::Is() const
    {
        try
        {
            fHolder->Type();
        }
        catch( XType* tObject )
        {
            return true;
        }
        catch( ... )
        {
            return false;
        }
        return false;
    }
    template< >
    inline bool KSObject::Is< KSObject >() const
    {
        return true;
    }

    template< class XType >
    inline XType* KSObject::As()
    {
        try
        {
            fHolder->Type();
        }
        catch( XType* tObject )
        {
            return tObject;
        }
        catch( ... )
        {
            return NULL;
        }
        return NULL;
    }
    template< >
    inline KSObject* KSObject::As< KSObject >()
    {
        return this;
    }

    template< class XType >
    inline const XType* KSObject::As() const
    {
        try
        {
            fHolder->Type();
        }
        catch( XType* tObject )
        {
            return tObject;
        }
        catch( ... )
        {
            return NULL;
        }
        return NULL;
    }
    template< >
    inline const KSObject* KSObject::As< KSObject >() const
    {
        return this;
    }

    template< class XType >
    inline void KSObject::Set( XType* anObject )
    {
        if( fHolder != NULL )
        {
            delete fHolder;
            fHolder = NULL;
        }
        KSHolderTemplate< XType >* tHolder = new KSHolderTemplate< XType >( anObject );
        fHolder = tHolder;
        return;
    }

}

#endif
