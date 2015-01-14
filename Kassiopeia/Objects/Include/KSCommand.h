#ifndef Kassiopeia_KSCommand_h_
#define Kassiopeia_KSCommand_h_

#include "KSObject.h"

namespace Kassiopeia
{

    class KSComponent;

    class KSCommand :
        public KSObject
    {
        public:
            KSCommand();
            KSCommand( const KSCommand& aCopy );
            virtual ~KSCommand();

        public:
            virtual KSCommand* Clone() const = 0;

        public:
            typedef enum
            {
                eIdle = 0,
                eActivated = 1
            } StateType;

            const StateType& State() const;

            void Activate();
            void Deactivate();

        protected:
            StateType fState;

            virtual void ActivateCommand();
            virtual void DeactivateCommand();

        public:
            void SetParent( KSComponent* aComponent );
            KSComponent* GetParent() const;

            void SetChild( KSComponent* aComponent );
            KSComponent* GetChild() const;

        protected:
            KSComponent* fParentComponent;
            KSComponent* fChildComponent;
    };

    template< >
    inline bool KSObject::Is< KSCommand >()
    {
        KSCommand* tCommand = dynamic_cast< KSCommand* >( this );
        if( tCommand != NULL )
        {
            return true;
        }
        return false;
    }

    template< >
    inline bool KSObject::Is< KSCommand >() const
    {
        const KSCommand* tCommand = dynamic_cast< const KSCommand* >( this );
        if( tCommand != NULL )
        {
            return true;
        }
        return false;
    }

    template< >
    inline KSCommand* KSObject::As< KSCommand >()
    {
        KSCommand* tCommand = dynamic_cast< KSCommand* >( this );
        if( tCommand != NULL )
        {
            return tCommand;
        }
        return NULL;
    }

    template< >
    inline const KSCommand* KSObject::As< KSCommand >() const
    {
        const KSCommand* tCommand = dynamic_cast< const KSCommand* >( this );
        if( tCommand != NULL )
        {
            return tCommand;
        }
        return NULL;
    }

}

#endif
