#ifndef Kassiopeia_KSComponentMinimumAt_h_
#define Kassiopeia_KSComponentMinimumAt_h_

#include "KSDictionary.h"
#include "KSNumerical.h"
#include "KSComponentValue.h"

namespace Kassiopeia
{

    template< class XValueType, class XValueTypeSource >
    class KSComponentMinimumAt :
        public KSComponent
    {
        public:
            KSComponentMinimumAt( KSComponent* aParentComponent, XValueType* aParentPointer, XValueTypeSource* aSourcePointer ) :
                    KSComponent(),
                    fMinimum( aSourcePointer ),
                    fOperand( aParentPointer ),
                    fMinimumAt( KSNumerical< XValueType >::Zero() )
            {
                Set( &fMinimumAt );
                this->SetParent( aParentComponent );
                aParentComponent->AddChild( this );
            }
            KSComponentMinimumAt( const KSComponentMinimumAt< XValueType, XValueTypeSource >& aCopy ) :
                    KSComponent( aCopy ),
                    fMinimum( aCopy.fMinimum ),
                    fOperand( aCopy.fOperand ),
                    fMinimumAt( aCopy.fMinimumAt )
            {
                Set( &fMinimumAt );
                this->SetParent( aCopy.fParentComponent );
                aCopy.fParentComponent->AddChild( this );
            }
            virtual ~KSComponentMinimumAt()
            {
            }

            //***********
            //KSComponent
            //***********

        public:
            KSComponent* Clone() const
            {
                return new KSComponentMinimumAt< XValueType, XValueTypeSource >( *this );
            }
            KSComponent* Component( const std::string& aField )
            {
                objctmsg_debug( "component minimum_at <" << this->GetName() << "> building component named <" << aField << ">" << eom )
                KSComponent* tComponent = KSDictionary< XValueType >::GetComponent( this, aField );
                if( tComponent == NULL )
                {
                    objctmsg( eError ) << "component minimum_at <" << this->GetName() << "> has no output named <" << aField << ">" << eom;
                }
                else
                {
                    fChildComponents.push_back( tComponent );
                }
                return tComponent;
            }
            KSCommand* Command( const std::string& /*aField*/, KSComponent* /*aChild*/ )
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
                objctmsg_debug( "component minimum_at <" << this->GetName() << "> pushing update" << eom );
                if ( fMinimum.Update() == true )
                {
                    fMinimumAt = (*fOperand);
                }
            }

            void PullDeupdateComponent()
            {
                objctmsg_debug( "component minimum_at <" << this->GetName() << "> pulling deupdate" << eom );
                fMinimum.Reset();
            }

        private:
            KSComponentValueMinimum< XValueTypeSource > fMinimum;
            XValueType* fOperand;
            XValueType fMinimumAt;
    };

}

#endif
