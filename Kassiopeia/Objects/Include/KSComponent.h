#ifndef Kassiopeia_KSComponent_h_
#define Kassiopeia_KSComponent_h_

#include "KSObject.h"

namespace Kassiopeia
{

    class KSCommand;

    class KSComponent :
        public KSObject
    {
        public:
            KSComponent();
            KSComponent( const KSComponent& aCopy );
            virtual ~KSComponent();

        public:
            virtual KSComponent* Clone() const = 0;
            virtual KSComponent* Component( const string& aField ) = 0;
            virtual KSCommand* Command( const string& aField, KSComponent* aChild ) = 0;

        public:
            typedef enum
            {
                eIdle = 0,
                eInitialized = 1,
                eActivated = 2,
                eUpdated = 3,
            } StateType;

            const StateType& State() const;

            void Initialize();
            void Deinitialize();
            void Activate();
            void Deactivate();
            void PushUpdate();
            void PushDeupdate();
            void PullUpdate();
            void PullDeupdate();

        protected:
            StateType fState;

            virtual void InitializeComponent();
            virtual void DeinitializeComponent();
            virtual void ActivateComponent();
            virtual void DeactivateComponent();
            virtual void PushUpdateComponent();
            virtual void PushDeupdateComponent();
            virtual void PullUpdateComponent();
            virtual void PullDeupdateComponent();

        public:
            void SetParent( KSComponent* aParent );
            KSComponent* GetParent() const;

            void AddChild( KSComponent* aChild );
            unsigned int GetChildCount() const;
            KSComponent* GetChild( const unsigned int& anIndex ) const;

        protected:
            KSComponent* fParentComponent;
            vector< KSComponent* > fChildComponents;
    };

    template< >
    inline bool KSObject::Is< KSComponent >()
    {
        KSComponent* tComponent = dynamic_cast< KSComponent* >( this );
        if( tComponent != NULL )
        {
            return true;
        }
        return false;
    }

    template< >
    inline bool KSObject::Is< KSComponent >() const
    {
        const KSComponent* tComponent = dynamic_cast< const KSComponent* >( this );
        if( tComponent != NULL )
        {
            return true;
        }
        return false;
    }

    template< >
    inline KSComponent* KSObject::As< KSComponent >()
    {
        KSComponent* tComponent = dynamic_cast< KSComponent* >( this );
        if( tComponent != NULL )
        {
            return tComponent;
        }
        return NULL;
    }

    template< >
    inline const KSComponent* KSObject::As< KSComponent >() const
    {
        const KSComponent* tComponent = dynamic_cast< const KSComponent* >( this );
        if( tComponent != NULL )
        {
            return tComponent;
        }
        return NULL;
    }

}

#endif
