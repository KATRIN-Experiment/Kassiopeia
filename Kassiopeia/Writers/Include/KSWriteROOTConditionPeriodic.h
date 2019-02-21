#ifndef Kassiopeia_KSWriteROOTConditionPeriodic_h_
#define Kassiopeia_KSWriteROOTConditionPeriodic_h_

#include "KSComponent.h"
#include "KSWriteROOTCondition.h"
#include "KField.h"
#include <limits>

namespace Kassiopeia
{

    template< class XValueType >
    class KSWriteROOTConditionPeriodic :
        public KSComponentTemplate< KSWriteROOTConditionPeriodic<XValueType>, KSWriteROOTCondition >
    {
        public:
            KSWriteROOTConditionPeriodic() :
                fInitialMin( -1.0*std::numeric_limits< XValueType >::max() ), //NOTE: defaults to always-on after reset
                fInitialMax( std::numeric_limits<XValueType>::max() ),
                fIncrement( 0. ),
                fResetMin( 0. ),
                fResetMax( 0. ),
                fComponent( 0 ),
                fValue( 0 ),
                fDone( false ),
                fMinValue( std::numeric_limits< XValueType >::max() ), //NOTE: defaults to always-off until reset
                fMaxValue( -1.0*std::numeric_limits<XValueType>::max() )
            {
			}
            KSWriteROOTConditionPeriodic( const KSWriteROOTConditionPeriodic& aCopy ) :
		            KSComponent(aCopy),
                    fInitialMin( aCopy.fInitialMin ),
                    fInitialMax( aCopy.fInitialMax ),
                    fIncrement( aCopy.fIncrement ),
                    fResetMin( aCopy.fResetMin ),
    				fResetMax( aCopy.fResetMax ),
    				fComponent( aCopy.fComponent ),
    				fValue( aCopy.fValue ),
                    fDone( aCopy.fDone ),
                    fMinValue( aCopy.fMinValue ),
    				fMaxValue( aCopy.fMaxValue )
            {
    	    }
            KSWriteROOTConditionPeriodic* Clone() const
    	    {
    	        return new KSWriteROOTConditionPeriodic( *this );
    	    }
    		virtual ~KSWriteROOTConditionPeriodic()
    		{
    		}

            void CalculateWriteCondition( bool& aFlag )
            {
                if (*fValue >= fResetMin && *fValue <= fResetMax )  // NOTE: this needs to occur even before the first use
                {
                    fDone = false;
                    fMinValue = fInitialMin;
                    fMaxValue = fInitialMax;
                }

        		if ( *fValue >= fMaxValue || *fValue <= fMinValue )
        		{
        			aFlag = false;
        			return;
        		}

                aFlag = true;

                fDone = true;

                fMinValue = fMinValue + fIncrement;
                fMaxValue = fMaxValue + fIncrement;

                return;
            }


        protected:
            ;K_SET_GET(double, InitialMin);
            ;K_SET_GET(double, InitialMax);
            ;K_SET_GET(double, Increment);
            ;K_SET_GET(double, ResetMin);
            ;K_SET_GET(double, ResetMax);
            ;K_SET_GET_PTR(KSComponent, Component);
            ;K_SET_GET_PTR( XValueType, Value );

        private:
            bool fDone;
            double fMinValue;
            double fMaxValue;

    };

}

#endif
