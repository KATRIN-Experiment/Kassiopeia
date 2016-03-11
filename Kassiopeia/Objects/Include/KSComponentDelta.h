#ifndef Kassiopeia_KSComponentDelta_h_
#define Kassiopeia_KSComponentDelta_h_

#include "KSDictionary.h"
#include "KSNumerical.h"
#include "KSComponentValue.h"

namespace Kassiopeia
{

    template< class XValueType >
    class KSComponentDelta :
        public KSComponent
    {
        public:
            KSComponentDelta( KSComponent* aParentComponent, XValueType* aParentPointer ) :
                    KSComponent(),
                    fDelta( aParentPointer )
            {
                Set( &fDelta );
                this->SetParent( aParentComponent );
                aParentComponent->AddChild( this );
            }
            KSComponentDelta( const KSComponentDelta< XValueType >& aCopy ) :
                    KSComponent( aCopy ),
                    fDelta( aCopy.fDelta )
            {
                Set( &fDelta );
                this->SetParent( aCopy.fParentComponent );
                aCopy.fParentComponent->AddChild( this );
            }
            virtual ~KSComponentDelta()
            {
            }

            //***********
            //KSComponent
            //***********

        public:
            KSComponent* Clone() const
            {
                return new KSComponentDelta< XValueType >( *this );
            }
            KSComponent* Component( const string& aField )
            {
                objctmsg_debug( "component delta <" << this->GetName() << "> building component named <" << aField << ">" << eom )
                KSComponent* tComponent = KSDictionary< XValueType >::GetComponent( this, aField );
                if( tComponent == NULL )
                {
                    objctmsg( eError ) << "component delta <" << this->GetName() << "> has no output named <" << aField << ">" << eom;
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
            void InitializeComponent()
            {
                fDelta.Reset();
            }

            void PushUpdateComponent()
            {
                objctmsg_debug( "component delta <" << this->GetName() << "> pushing update" << eom );
                (void) fDelta.Update();
                return;
            }

            void PullDeupdateComponent()
            {
                objctmsg_debug( "component delta <" << this->GetName() << "> pulling deupdate" << eom );
                fDelta.Reset();
                return;
            }

        private:
            KSComponentValueDelta< XValueType > fDelta;
    };

}

#endif
