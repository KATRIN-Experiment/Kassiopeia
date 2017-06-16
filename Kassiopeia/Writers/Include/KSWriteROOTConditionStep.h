#ifndef Kassiopeia_KSWriteROOTConditionStep_h_
#define Kassiopeia_KSWriteROOTConditionStep_h_

#include "KSComponent.h"
#include "KSWriteROOTCondition.h"
#include "KField.h"
#include <limits>

namespace Kassiopeia
{

    template< class XValueType >
    class KSWriteROOTConditionStep :
        public KSComponentTemplate< KSWriteROOTConditionStep<XValueType>, KSWriteROOTCondition >
    {
        public:
            KSWriteROOTConditionStep() :
                fNthStepValue( 1 ),
                fComponent( 0 ),
                fValue( 0 )
            {
			}
            KSWriteROOTConditionStep( const KSWriteROOTConditionStep& aCopy ) :
		            KSComponent(aCopy),
    				fNthStepValue( aCopy.fNthStepValue ),
    				fComponent( aCopy.fComponent ),
    				fValue( aCopy.fValue )
            {
    	    }
            KSWriteROOTConditionStep* Clone() const
    	    {
    	        return new KSWriteROOTConditionStep( *this );
    	    }
    		virtual ~KSWriteROOTConditionStep()
    		{
    		}

            void CalculateWriteCondition( bool& aFlag )
            {
        		if ( *fValue % fNthStepValue == 0 )
        		{
        			aFlag = true;
        			return;
        		}

                aFlag = false;
                return;
            }


        protected:
            ;K_SET_GET(int, NthStepValue);
            ;K_SET_GET_PTR(KSComponent, Component);
            ;K_SET_GET_PTR( XValueType, Value );

    };

}

#endif
