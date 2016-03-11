#ifndef Kassiopeia_KSTrajTermGyration_h_
#define Kassiopeia_KSTrajTermGyration_h_

#include "KSComponentTemplate.h"
#include "KSTrajAdiabaticTypes.h"

namespace Kassiopeia
{

    class KSTrajTermGyration :
        public KSComponentTemplate< KSTrajTermGyration >,
        public KSTrajAdiabaticDifferentiator
    {
        public:
            KSTrajTermGyration();
            KSTrajTermGyration( const KSTrajTermGyration& aCopy );
            KSTrajTermGyration* Clone() const;
            virtual ~KSTrajTermGyration();

        public:
            virtual void Differentiate(double /*aTime*/, const KSTrajAdiabaticParticle& aParticle, KSTrajAdiabaticDerivative& aDerivative ) const;
    };

}

#endif
