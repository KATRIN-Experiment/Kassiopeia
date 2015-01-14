#ifndef KSGENCREATOR_H
#define KSGENCREATOR_H

#include "KSComponentTemplate.h"
#include "KSParticle.h"

#include <vector>
using std::vector;

namespace Kassiopeia
{

    class KSGenCreator :
        public KSComponentTemplate< KSGenCreator >
    {
        public:
            KSGenCreator();
            virtual ~KSGenCreator();

        public:
            virtual void Dice( KSParticleQueue* aPrimaries ) = 0;
    };

}

#endif // KSGENCREATOR_H
