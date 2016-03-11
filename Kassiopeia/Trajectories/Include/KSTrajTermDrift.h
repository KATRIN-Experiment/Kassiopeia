#ifndef Kassiopeia_KSTrajTermDrift_h_
#define Kassiopeia_KSTrajTermDrift_h_

#include "KSComponentTemplate.h"
#include "KSTrajAdiabaticTypes.h"

namespace Kassiopeia
{

    class KSTrajTermDrift :
        public KSComponentTemplate< KSTrajTermDrift >,
        public KSTrajAdiabaticDifferentiator
    {
        public:
            KSTrajTermDrift();
            KSTrajTermDrift( const KSTrajTermDrift& aCopy );
            KSTrajTermDrift* Clone() const;
            virtual ~KSTrajTermDrift();

        public:
            virtual void Differentiate(double /*aTime*/, const KSTrajAdiabaticParticle& aParticle, KSTrajAdiabaticDerivative& aDerivative ) const;
    };

}

#endif
