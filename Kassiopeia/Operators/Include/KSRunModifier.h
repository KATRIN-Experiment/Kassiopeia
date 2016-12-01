#ifndef Kassiopeia_KSRunModifier_h_
#define Kassiopeia_KSRunModifier_h_

#include "KSComponentTemplate.h"

namespace Kassiopeia
{
    class KSRun;

    class KSRunModifier:
            public KSComponentTemplate< KSRunModifier >
    {
        public:
            KSRunModifier();
            virtual ~KSRunModifier();

        public:

            //returns true if any of the state variables of anRun are changed
            virtual bool ExecutePreRunModification( KSRun& aRun ) = 0;

            //returns true if any of the state variables of anRun are changed
            virtual bool ExecutePostRunModification( KSRun& aRun ) = 0;
    };

}

#endif
