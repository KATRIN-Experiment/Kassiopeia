#ifndef Kassiopeia_KSComponentTemplate_h_
#define Kassiopeia_KSComponentTemplate_h_

#include "KSComponent.h"
#include "KSComponentMember.h"
#include "KSCommandMember.h"

namespace Kassiopeia
{

    template< class XThisType, class XParentOne = void, class XParentTwo = void, class XParentThree = void >
    class KSComponentTemplate;

    //******************
    //1-parent component
    //******************

    template< class XThisType, class XFirstParentType >
    class KSComponentTemplate< XThisType, XFirstParentType, void, void > :
        virtual public KSComponent,
        public XFirstParentType
    {
        public:
            KSComponentTemplate()
            {
                Set( static_cast< XThisType* >( this ) );
            }
            virtual ~KSComponentTemplate()
            {
            }

            //***********
            //KSComponent
            //***********

        public:
            KSComponent* Component( const string& aField )
            {
                objctmsg_debug( "component <" << this->GetName() << "> building output named <" << aField << ">" << eom )
                KSComponent* tComponent = KSDictionary< XThisType >::GetComponent( this, aField );
                if( tComponent == NULL )
                {
                    return XFirstParentType::Component( aField );
                }
                else
                {
                    fChildComponents.push_back( tComponent );
                    return tComponent;
                }
            }
            KSCommand* Command( const string& aField, KSComponent* aChild )
            {
                objctmsg_debug( "component <" << this->GetName() << "> building command named <" << aField << ">" << eom )
                KSCommand* tCommand = KSDictionary< XThisType >::GetCommand( this, aChild, aField );
                if( tCommand == NULL )
                {
                    return XFirstParentType::Command( aField, aChild );
                }
                else
                {
                    return tCommand;
                }
            }
    };

    //******************
    //0-parent component
    //******************

    template< class XThisType >
    class KSComponentTemplate< XThisType, void, void, void > :
        virtual public KSComponent
    {
        public:
            KSComponentTemplate() :
                    KSComponent()
            {
                Set( static_cast< XThisType* >( this ) );
            }
            virtual ~KSComponentTemplate()
            {
            }

            //***********
            //KSComponent
            //***********

        public:
            KSComponent* Component( const string& aLabel )
            {
                objctmsg_debug( "component <" << this->GetName() << "> building component named <" << aLabel << ">" << eom )
                KSComponent* tComponent = KSDictionary< XThisType >::GetComponent( this, aLabel );
                if( tComponent == NULL )
                {
                    objctmsg( eError ) << "component <" << this->GetName() << "> has no component named <" << aLabel << ">" << eom;
                    return NULL;
                }
                else
                {
                    fChildComponents.push_back( tComponent );
                    return tComponent;
                }
            }
            KSCommand* Command( const string& aField, KSComponent* aChild )
            {
                objctmsg_debug( "component <" << this->GetName() << "> building command named <" << aField << ">" << eom )
                KSCommand* tCommand = KSDictionary< XThisType >::GetCommand( this, aChild, aField );
                if( tCommand == NULL )
                {
                    objctmsg( eError ) << "component <" << this->GetName() << "> has no command named <" << aField << ">" << eom;
                    return NULL;
                }
                else
                {
                    return tCommand;
                }
            }
    };

}

#endif
