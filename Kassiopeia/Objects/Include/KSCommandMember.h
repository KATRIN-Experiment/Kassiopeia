#ifndef Kassiopeia_KSCommandTemplate_h_
#define Kassiopeia_KSCommandTemplate_h_

#include "KSDictionary.h"
#include <typeinfo>

namespace Kassiopeia
{

    template< class XParentType, class XChildType >
    class KSCommandMemberAdd :
        public KSCommand
    {
        public:
            KSCommandMemberAdd( KSComponent* aParentComponent, XParentType* aParent, KSComponent* aChildComponent, XChildType* aChild, void (XParentType::*anAddMember)( XChildType* ), void (XParentType::*aRemoveMember)( XChildType* ) ) :
                    KSCommand(),
                    fParentPointer( aParent ),
                    fChildPointer( aChild ),
                    fAddMember( anAddMember ),
                    fRemoveMember( aRemoveMember )
            {
                Set( this );
                fParentComponent = aParentComponent;
                fChildComponent = aChildComponent;
            }
            KSCommandMemberAdd( const KSCommandMemberAdd< XParentType, XChildType >& aCopy ) :
                    KSCommand( aCopy ),
                    fParentPointer( aCopy.fParentPointer ),
                    fChildPointer( aCopy.fChildPointer ),
                    fAddMember( aCopy.fAddMember ),
                    fRemoveMember( aCopy.fRemoveMember )

            {
                Set( this );
                fParentComponent = aCopy.fParentComponent;
                fChildComponent = aCopy.fChildComponent;
            }
            virtual ~KSCommandMemberAdd()
            {
            }

        public:
            KSCommandMemberAdd* Clone() const
            {
                return new KSCommandMemberAdd< XParentType, XChildType >( *this );
            }

        protected:
            void ActivateCommand()
            {
                (fParentPointer->*fAddMember)( fChildPointer );
                fChildComponent->Activate();
                return;
            }
            void DeactivateCommand()
            {
                fChildComponent->Deactivate();
                (fParentPointer->*fRemoveMember)( fChildPointer );
                return;
            }

        private:
            XParentType* fParentPointer;
            XChildType* fChildPointer;
            void (XParentType::*fAddMember)( XChildType* );
            void (XParentType::*fRemoveMember)( XChildType* );
    };

    template< class XParentType, class XChildType >
    class KSCommandMemberAddFactory :
        public KSCommandFactory
    {
        public:
            KSCommandMemberAddFactory( void (XParentType::*anAddMember)( XChildType* ), void (XParentType::*aRemoveMember)( XChildType* ) ) :
                    fAddMember( anAddMember ),
                    fRemoveMember( aRemoveMember )
            {
            }
            virtual ~KSCommandMemberAddFactory()
            {
            }

        public:
            KSCommand* CreateCommand( KSComponent* aParent, KSComponent* aChild ) const
            {
                XParentType* tParent = aParent->As< XParentType >();
                if( tParent == NULL )
                {
                    objctmsg_debug( "  command parent <" << aParent->GetName() << "> could not be cast to type <" << typeid( XParentType ).name() << ">" << eom );
                    return NULL;
                }

                XChildType* tChild = aChild->As< XChildType >();
                if( tChild == NULL )
                {
                    objctmsg_debug( "  command child <" << aChild->GetName() << "> could not be cast to type <" << typeid( XChildType ).name() << ">" << eom );
                    return NULL;
                }

                objctmsg_debug( "  command built" << eom );
                return new KSCommandMemberAdd< XParentType, XChildType >( aParent, tParent, aChild, tChild, fAddMember, fRemoveMember );
            }

        private:
            void (XParentType::*fAddMember)( XChildType* );
            void (XParentType::*fRemoveMember)( XChildType* );
    };


    template< class XParentType, class XChildType >
    class KSCommandMemberRemove :
        public KSCommand
    {
        public:
            KSCommandMemberRemove( KSComponent* aParentComponent, XParentType* aParent, KSComponent* aChildComponent, XChildType* aChild, void (XParentType::*anAddMember)( XChildType* ), void (XParentType::*aRemoveMember)( XChildType* ) ) :
                    KSCommand(),
                    fParentPointer( aParent ),
                    fChildPointer( aChild ),
                    fAddMember( anAddMember ),
                    fRemoveMember( aRemoveMember )
            {
                Set( this );
                fParentComponent = aParentComponent;
                fChildComponent = aChildComponent;
            }
            KSCommandMemberRemove( const KSCommandMemberRemove< XParentType, XChildType >& aCopy ) :
                    KSCommand( aCopy ),
                    fParentPointer( aCopy.fParentPointer ),
                    fChildPointer( aCopy.fChildPointer ),
                    fAddMember( aCopy.fAddMember ),
                    fRemoveMember( aCopy.fRemoveMember )

            {
                Set( this );
                fParentComponent = aCopy.fParentComponent;
                fChildComponent = aCopy.fChildComponent;
            }
            virtual ~KSCommandMemberRemove()
            {
            }

        public:
            KSCommandMemberRemove* Clone() const
            {
                return new KSCommandMemberRemove< XParentType, XChildType >( *this );
            }

        protected:
            void ActivateCommand()
            {
                fChildComponent->Deactivate();
                (fParentPointer->*fRemoveMember)( fChildPointer );
                return;
            }
            void DeactivateCommand()
            {
                (fParentPointer->*fAddMember)( fChildPointer );
                fChildComponent->Activate();
                return;
            }

        private:
            XParentType* fParentPointer;
            XChildType* fChildPointer;
            void (XParentType::*fAddMember)( XChildType* );
            void (XParentType::*fRemoveMember)( XChildType* );
    };

    template< class XParentType, class XChildType >
    class KSCommandMemberRemoveFactory :
        public KSCommandFactory
    {
        public:
            KSCommandMemberRemoveFactory( void (XParentType::*anAddMember)( XChildType* ), void (XParentType::*aRemoveMember)( XChildType* ) ) :
                    fAddMember( anAddMember ),
                    fRemoveMember( aRemoveMember )
            {
            }
            virtual ~KSCommandMemberRemoveFactory()
            {
            }

        public:
            KSCommand* CreateCommand( KSComponent* aParent, KSComponent* aChild ) const
            {
                XParentType* tParent = aParent->As< XParentType >();
                if( tParent == NULL )
                {
                    objctmsg_debug( "  command parent <" << aParent->GetName() << "> could not be cast to type <" << typeid( XParentType ).name() << ">" << eom );
                    return NULL;
                }

                XChildType* tChild = aChild->As< XChildType >();
                if( tChild == NULL )
                {
                    objctmsg_debug( "  command child <" << aChild->GetName() << "> could not be cast to type <" << typeid( XChildType ).name() << ">" << eom );
                    return NULL;
                }

                objctmsg_debug( "  command built" << eom );
                return new KSCommandMemberRemove< XParentType, XChildType >( aParent, tParent, aChild, tChild, fAddMember, fRemoveMember );
            }

        private:
            void (XParentType::*fAddMember)( XChildType* );
            void (XParentType::*fRemoveMember)( XChildType* );
    };


    template< class XParentType, class XChildType >
    class KSCommandMemberParameter :
        public KSCommand
    {
        public:
            KSCommandMemberParameter( KSComponent* aParentComponent, XParentType* aParentPointer, KSComponent* aChildComponent, XChildType* aChildPointer, void (XParentType::*aSetMember)( const XChildType& ), const XChildType& (XParentType::*aGetMember)() const ) :
                    KSCommand(),
                    fParentPointer( aParentPointer ),
                    fChildPointer( aChildPointer ),
                    fSetMember( aSetMember ),
                    fGetMember( aGetMember )
            {
                Set( this );
                fParentComponent = aParentComponent;
                fChildComponent = aChildComponent;
            }
            KSCommandMemberParameter( const KSCommandMemberParameter< XParentType, XChildType >& aCopy ) :
                    KSCommand( aCopy ),
                    fParentPointer( aCopy.fParentPointer ),
                    fChildPointer( aCopy.fChildPointer ),
                    fSetMember( aCopy.fSetMember ),
                    fGetMember( aCopy.fGetMember )
            {
                Set( this );
                fParentComponent = aCopy.fParentComponent;
                fChildComponent = aCopy.fChildComponent;
            }
            virtual ~KSCommandMemberParameter()
            {
            }

        public:
            KSCommandMemberParameter* Clone() const
            {
                return new KSCommandMemberParameter< XParentType, XChildType >( *this );
            }

        protected:
            void ActivateCommand()
            {
                XChildType tOldValue = (fParentPointer->*fGetMember)();
                (fParentPointer->fSetMember)( *fChildPointer );
                (*fChildPointer) = tOldValue;
                return;
            }
            void DeactivateCommand()
            {
                XChildType tOldValue = (fParentPointer->*fGetMember)();
                (fParentPointer->fSetMember)( *fChildPointer );
                (*fChildPointer) = tOldValue;
                return;
            }

        private:
            XParentType* fParentPointer;
            XChildType* fChildPointer;
            void (XParentType::*fSetMember)( const XChildType& );
            const XChildType& (XParentType::*fGetMember)() const;
    };

    template< class XParentType, class XChildType >
    class KSCommandMemberParameterFactory :
        public KSCommandFactory
    {
        public:
            KSCommandMemberParameterFactory( void (XParentType::*aSetMember)( const XChildType& ), const XChildType& (XParentType::*aGetMember)() const ) :
                    fSetMember( aSetMember ),
                    fGetMember( aGetMember )
            {
            }
            virtual ~KSCommandMemberParameterFactory()
            {
            }

        public:
            KSCommand* CreateCommand( KSComponent* aParent, KSComponent* aChild ) const
            {
                XParentType* tParent = aParent->As< XParentType >();
                if( tParent == NULL )
                {
                    objctmsg_debug( "  command parent <" << aParent->GetName() << "> could not be cast to type <" << typeid( XParentType ).name() << ">" << eom );
                    return NULL;
                }

                XChildType* tChild = aChild->As< XChildType >();
                if( tChild == NULL )
                {
                    objctmsg_debug( "  command child <" << aParent->GetName() << "> could not be cast to type <" << typeid( XChildType ).name() << ">" << eom );
                    return NULL;
                }

                objctmsg_debug( "  command built" << eom );
                return new KSCommandMemberParameter< XParentType, XChildType >( aParent, tParent, aChild, tChild, fSetMember, fGetMember );
            }

        private:
            void (XParentType::*fSetMember)( const XChildType& );
            const XChildType& (XParentType::*fGetMember)() const;
    };


    template< class XType >
    template< class XParentType, class XChildType >
    int KSDictionary< XType >::AddCommand( void (XParentType::*anAddMember)( XChildType* ), void (XParentType::*aRemoveMember)( XChildType* ), const string& anAddField, const string& aRemoveField )
    {
        if( fCommandFactories == NULL )
        {
            fCommandFactories = new CommandFactoryMap();
        }

        KSCommandMemberAddFactory< XParentType, XChildType >* tAddFactory = new KSCommandMemberAddFactory< XParentType, XChildType >( anAddMember, aRemoveMember );
        fCommandFactories->insert( CommandFactoryEntry( anAddField, tAddFactory ) );

        KSCommandMemberRemoveFactory< XParentType, XChildType >* tRemoveFactory = new KSCommandMemberRemoveFactory< XParentType, XChildType >( anAddMember, aRemoveMember );
        fCommandFactories->insert( CommandFactoryEntry( aRemoveField, tRemoveFactory ) );

        return 0;
    }

    template< class XType >
    template< class XParentType, class XChildType >
    int KSDictionary< XType >::AddCommand( void (XParentType::*aSetMember)( const XChildType& ), const XChildType& (XParentType::*aGetMember)() const, const string& aParameterField )
    {
        if( fCommandFactories == NULL )
        {
            fCommandFactories = new CommandFactoryMap();
        }

        KSCommandMemberParameterFactory< XParentType, XChildType >* tParameterFactory = new KSCommandMemberParameterFactory< XParentType, XChildType >( aSetMember, aGetMember );
        fCommandFactories->insert( CommandFactoryEntry( aParameterField, tParameterFactory ) );

        return 0;
    }

}

#endif
