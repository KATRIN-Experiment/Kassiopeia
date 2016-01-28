#ifndef Kassiopeia_KSComponentMinimum_h_
#define Kassiopeia_KSComponentMinimum_h_

#include "KSDictionary.h"
#include "KSNumerical.h"
#include "KSComponentValue.h"

namespace Kassiopeia
{

    template< class XValueType >
    class KSComponentMinimum :
        public KSComponent
    {
        public:
            KSComponentMinimum( KSComponent* aParentComponent, XValueType* aParentPointer ) :
                    KSComponent(),
                    fMinimum( aParentPointer )
            {
                Set( &fMinimum );
                this->SetParent( aParentComponent );
                aParentComponent->AddChild( this );
            }
            KSComponentMinimum( const KSComponentMinimum< XValueType >& aCopy ) :
                    KSComponent( aCopy ),
                    fMinimum( aCopy.fMinimum )
            {
                Set( &fMinimum );
                this->SetParent( aCopy.fParentComponent );
                aCopy.fParentComponent->AddChild( this );
            }
            virtual ~KSComponentMinimum()
            {
            }

            //***********
            //KSComponent
            //***********

        public:
            KSComponent* Clone() const
            {
                return new KSComponentMinimum< XValueType >( *this );
            }
            KSComponent* Component( const string& aField )
            {
                objctmsg_debug( "component minimum <" << this->GetName() << "> building component named <" << aField << ">" << eom )
                KSComponent* tComponent = KSDictionary< XValueType >::GetComponent( this, aField );
                if( tComponent == NULL )
                {
                    objctmsg( eError ) << "component minimum <" << this->GetName() << "> has no output named <" << aField << ">" << eom;
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
                fMinimum.Reset();
            }

            void PushUpdateComponent()
            {
                objctmsg_debug( "component minimum <" << this->GetName() << "> pushing update" << eom );
                (void) fMinimum.Update();
            }

            void PullDeupdateComponent()
            {
                objctmsg_debug( "component minimum <" << this->GetName() << "> pulling deupdate" << eom );
                fMinimum.Reset();
            }

        private:
            KSComponentValueMinimum< XValueType > fMinimum;
    };

}

#endif
