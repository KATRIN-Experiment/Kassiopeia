#ifndef Kassiopeia_KSEventModifier_h_
#define Kassiopeia_KSEventModifier_h_

#include "KSComponentTemplate.h"

namespace Kassiopeia
{

    class KSEvent;

    class KSEventModifier:
            public KSComponentTemplate< KSEventModifier >
    {
        public:
            KSEventModifier();
            virtual ~KSEventModifier();

        public:

            //returns true if any of the state variables of anInitialParticle are changed
            virtual bool ExecutePreEventModification() = 0;

            //returns true if any of the state variables of aFinalParticle are changed
            virtual bool ExecutePostEventModifcation() = 0;
    };

}

#endif
