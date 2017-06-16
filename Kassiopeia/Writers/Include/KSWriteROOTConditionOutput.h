#ifndef Kassiopeia_KSWriteROOTConditionOutput_h_
#define Kassiopeia_KSWriteROOTConditionOutput_h_

#include "KSComponent.h"
#include "KSWriteROOTCondition.h"
#include "KField.h"
#include <limits>

namespace Kassiopeia
{

    template< class XValueType >
    class KSWriteROOTConditionOutput :
        public KSComponentTemplate< KSWriteROOTConditionOutput<XValueType>, KSWriteROOTCondition >
    {
        public:
            KSWriteROOTConditionOutput() :
                fMinValue( -1.0*std::numeric_limits< XValueType >::max() ),
                fMaxValue( std::numeric_limits<XValueType>::max() ),
                fComponent( 0 ),
                fValue( 0 )
            {
			}
            KSWriteROOTConditionOutput( const KSWriteROOTConditionOutput& aCopy ) :
		            KSComponent(aCopy),
    				fMinValue( aCopy.fMinValue ),
    				fMaxValue( aCopy.fMaxValue ),
    				fComponent( aCopy.fComponent ),
    				fValue( aCopy.fValue )
            {
    	    }
            KSWriteROOTConditionOutput* Clone() const
    	    {
    	        return new KSWriteROOTConditionOutput( *this );
    	    }
    		virtual ~KSWriteROOTConditionOutput()
    		{
    		}

            void CalculateWriteCondition( bool& aFlag )
            {
        		if ( *fValue >= fMaxValue || *fValue <= fMinValue )
        		{
        			aFlag = false;
        			return;
        		}

                aFlag = true;
                return;
            }


        protected:
            ;K_SET_GET(double, MinValue);
            ;K_SET_GET(double, MaxValue);
            ;K_SET_GET_PTR(KSComponent, Component);
            ;K_SET_GET_PTR( XValueType, Value );

    };

}

#endif
