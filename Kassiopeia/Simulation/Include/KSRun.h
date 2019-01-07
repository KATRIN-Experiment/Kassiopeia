#ifndef Kassiopeia_KSRun_h_
#define Kassiopeia_KSRun_h_

#include "KSComponentTemplate.h"
#include "KField.h"

namespace Kassiopeia
{

    class KSRun :
        public KSComponentTemplate< KSRun >
    {
        public:
            KSRun();
            KSRun( const KSRun& aCopy );
            KSRun& operator=( const KSRun& aCopy );
            KSRun* Clone() const;
            ~KSRun();

            //***
            //IDs
            //***

            K_REFS( int, RunId )
            K_REFS( int, RunCount )

            //***
            //run
            //***

            K_REFS( unsigned int, TotalEvents )
            K_REFS( unsigned int, TotalTracks )
            K_REFS( unsigned int, TotalSteps )
            K_REFS( double, ContinuousTime )
            K_REFS( double, ContinuousLength )
            K_REFS( double, ContinuousEnergyChange )
            K_REFS( double, ContinuousMomentumChange )
            K_REFS( unsigned int, DiscreteSecondaries )
            K_REFS( double, DiscreteEnergyChange )
            K_REFS( double, DiscreteMomentumChange )
            K_REFS( unsigned int, NumberOfTurns )
    };

}

#endif
