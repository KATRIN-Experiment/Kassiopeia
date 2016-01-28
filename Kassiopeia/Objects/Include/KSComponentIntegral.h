#ifndef Kassiopeia_KSComponentIntegral_h_
#define Kassiopeia_KSComponentIntegral_h_

#include "KSDictionary.h"
#include "KSNumerical.h"
#include "KSComponentValue.h"

namespace Kassiopeia
{

    template< class XValueType >
    class KSComponentIntegral :
        public KSComponent
    {
        public:
            KSComponentIntegral( KSComponent* aParentComponent, XValueType* aParentPointer ) :
                    KSComponent(),
                    fIntegral( aParentPointer )
            {
                Set( &fIntegral );
                this->SetParent( aParentComponent );
                aParentComponent->AddChild( this );
            }
            KSComponentIntegral( const KSComponentIntegral< XValueType >& aCopy ) :
                    KSComponent( aCopy ),
                    fIntegral( aCopy.fIntegral )
            {
                Set( &fIntegral );
                this->SetParent( aCopy.fParentComponent );
                aCopy.fParentComponent->AddChild( this );
            }
            virtual ~KSComponentIntegral()
            {
            }

            //***********
            //KSComponent
            //***********

        public:
            KSComponent* Clone() const
            {
                return new KSComponentIntegral< XValueType >( *this );
            }
            KSComponent* Component( const string& aField )
            {
                objctmsg_debug( "component integral <" << this->GetName() << "> building component named <" << aField << ">" << eom )
                KSComponent* tComponent = KSDictionary< XValueType >::GetComponent( this, aField );
                if( tComponent == NULL )
                {
                    objctmsg( eError ) << "component integral <" << this->GetName() << "> has no output named <" << aField << ">" << eom;
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
                fIntegral.Reset();
            }

            void PushUpdateComponent()
            {
                objctmsg_debug( "component integral <" << this->GetName() << "> pushing update" << eom );
                (void) fIntegral.Update();
                return;
            }

            void PullDeupdateComponent()
            {
                objctmsg_debug( "component integral <" << this->GetName() << "> pulling deupdate" << eom );
                fIntegral.Reset();
                return;
            }

        private:
            KSComponentValueIntegral< XValueType > fIntegral;
    };

}

#endif
