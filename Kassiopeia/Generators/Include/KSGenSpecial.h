#ifndef Kassiopeia_KSGenSpecial_h_
#define Kassiopeia_KSGenSpecial_h_

#include "KSComponentTemplate.h"
#include "KSParticle.h"

#include <vector>
using std::vector;

namespace Kassiopeia
{

    class KSGenSpecial :
        public KSComponentTemplate< KSGenSpecial >
    {
        public:
            KSGenSpecial();
            virtual ~KSGenSpecial();

        public:
            virtual void DiceSpecial( KSParticleQueue* aPrimaries ) = 0;
    };

}

#endif
