#ifndef Kassiopeia_KSComponentMaximumAt_h_
#define Kassiopeia_KSComponentMaximumAt_h_

#include "KSDictionary.h"
#include "KSNumerical.h"
#include "KSComponentValue.h"

namespace Kassiopeia
{
    template< class XValueType, class XValueTypeSource >
    class KSComponentMaximumAt :
        public KSComponent
    {
        public:
            KSComponentMaximumAt( KSComponent* aParentComponent, XValueType* aParentPointer, XValueTypeSource* aSourcePointer ) :
                    KSComponent(),
                    fMaximum( aSourcePointer ),
                    fOperand( aParentPointer ),
                    fMaximumAt( KSNumerical< XValueType >::Zero() )
            {
                Set( &fMaximumAt );
                this->SetParent( aParentComponent );
                aParentComponent->AddChild( this );
            }
            KSComponentMaximumAt( const KSComponentMaximumAt< XValueType, XValueTypeSource >& aCopy ) :
                    KSComponent( aCopy ),
                    fMaximum( aCopy.fMaximum ),
                    fOperand( aCopy.fOperand ),
                    fMaximumAt( aCopy.fMaximumAt )
            {
                Set( &fMaximumAt );
                this->SetParent( aCopy.fParentComponent );
                aCopy.fParentComponent->AddChild( this );
            }
            virtual ~KSComponentMaximumAt()
            {
            }

            //***********
            //KSComponent
            //***********

        public:
            KSComponent* Clone() const
            {
                return new KSComponentMaximumAt< XValueType, XValueTypeSource >( *this );
            }
            KSComponent* Component( const std::string& aField )
            {
                objctmsg_debug( "component maximum_at <" << this->GetName() << "> building component named <" << aField << ">" << eom )
                KSComponent* tComponent = KSDictionary< XValueType >::GetComponent( this, aField );
                if( tComponent == NULL )
                {
                    objctmsg( eError ) << "component maximum_at <" << this->GetName() << "> has no output named <" << aField << ">" << eom;
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
                fMaximum.Reset();
            }

            void PushUpdateComponent()
            {
                objctmsg_debug( "component maximum_at <" << this->GetName() << "> pushing update" << eom );
                if ( fMaximum.Update() == true )
                {
                    fMaximumAt = (*fOperand);
                }
            }

            void PullDeupdateComponent()
            {
                objctmsg_debug( "component maximum_at <" << this->GetName() << "> pulling deupdate" << eom );
                fMaximum.Reset();
            }

        private:
            KSComponentValueMaximum< XValueTypeSource > fMaximum;
            XValueType* fOperand;
            XValueType fMaximumAt;
    };

}

#endif
