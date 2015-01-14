#ifndef Kassiopeia_KSComponentMaximum_h_
#define Kassiopeia_KSComponentMaximum_h_

#include "KSDictionary.h"
#include "KSNumerical.h"
#include <iostream>
namespace Kassiopeia
{

    template< class XValueType >
    class KSComponentMaximum :
        public KSComponent
    {
        public:
            KSComponentMaximum( KSComponent* aParentComponent, XValueType* aParentPointer ) :
                    KSComponent(),
                    fOperand( aParentPointer ),
                    fMaximum( -1. * KSNumerical< XValueType >::Maximum )
            {
                Set( &fMaximum );
                this->SetParent( aParentComponent );
                aParentComponent->AddChild( this );
            }
            KSComponentMaximum( const KSComponentMaximum< XValueType >& aCopy ) :
                    KSComponent( aCopy ),
                    fOperand( aCopy.fOperand ),
                    fMaximum( aCopy.fMaximum )
            {
                Set( &fMaximum );
                this->SetParent( aCopy.fParentComponent );
                aCopy.fParentComponent->AddChild( this );
            }
            virtual ~KSComponentMaximum()
            {
            }

            //***********
            //KSComponent
            //***********

        public:
            KSComponent* Clone() const
            {
                return new KSComponentMaximum< XValueType >( *this );
            }
            KSComponent* Component( const string& aField )
            {
                objctmsg_debug( "component maximum <" << this->GetName() << "> building component named <" << aField << ">" << eom )
                KSComponent* tComponent = KSDictionary< XValueType >::GetComponent( this, aField );
                if( tComponent == NULL )
                {
                    objctmsg( eError ) << "component maximum <" << this->GetName() << "> has no output named <" << aField << ">" << eom;
                }
                else
                {
                    fChildComponents.push_back( tComponent );
                }
                return tComponent;
            }
            KSCommand* Command( const string& /*aField*/, KSComponent* /*aChild*/ )
            {
                return NULL;
            }

        public:
            void PushUpdateComponent()
            {
                objctmsg_debug( "component maximum <" << this->GetName() << "> pushing update" << eom );
                if( fMaximum < (*fOperand) )
                {
                    fMaximum = (*fOperand);
                }
                return;
            }

            void PullDeupdateComponent()
            {
                objctmsg_debug( "component maximum <" << this->GetName() << "> pulling deupdate" << eom );
                fMaximum = -1. * KSNumerical< XValueType >::Maximum;
                return;
            }

        private:
            XValueType* fOperand;
            XValueType fMaximum;
    };

}

#endif
