#ifndef Kassiopeia_KSTermOutput_h_
#define Kassiopeia_KSTermOutput_h_

#include "KSTerminator.h"
#include "KSComponent.h"
#include "KSParticle.h"
#include "KField.h"
#include <limits>

namespace Kassiopeia
{

    template< class XValueType >
    class KSTermOutput :
        public KSComponentTemplate< KSTermOutput<XValueType>, KSTerminator >
    {
        public:
    		KSTermOutput() :
                fMinValue( -1.0*std::numeric_limits< XValueType >::max() ),
                fMaxValue( std::numeric_limits<XValueType>::max() ),
                fValue( 0 ),
                fFirstStep( true )
			{
			}
    		KSTermOutput( const KSTermOutput& aCopy ) :
		            KSComponent(aCopy),
    				fMinValue( aCopy.fMinValue ),
    				fMaxValue( aCopy.fMaxValue ),
    				fValue( aCopy.fValue ),
    				fFirstStep( aCopy.fFirstStep )
    		{
    	    }
    		KSTermOutput* Clone() const
    	    {
    	        return new KSTermOutput( *this );
    	    }
    		virtual ~KSTermOutput()
    		{
    		}

            void CalculateTermination( const KSParticle& /*anInitialParticle*/, bool& aFlag )
            {
				if ( fFirstStep == true )
				{
					fFirstStep = false;
	                aFlag = false;
					return;
				}

        		if ( *fValue >= fMaxValue || *fValue <= fMinValue )
        		{
        			aFlag = true;
        			return;
        		}

                aFlag = false;
                return;
            }

            void ExecuteTermination( const KSParticle& /*anInitialParticle*/, KSParticle& aFinalParticle, KSParticleQueue& /*aParticleQueue*/ ) const
            {
                aFinalParticle.SetActive( false );
                aFinalParticle.SetLabel( katrin::KNamed::GetName() );
                return;
            }

            virtual void ActivateComponent()
            {
            	fFirstStep = true;
            }

        protected:
            ;K_SET_GET(double, MinValue);
            ;K_SET_GET(double, MaxValue);
            ;K_SET_GET_PTR( XValueType, Value );
            bool fFirstStep;

    };

}

#endif
