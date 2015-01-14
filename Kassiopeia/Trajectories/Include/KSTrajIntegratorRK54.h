#ifndef Kassiopeia_KSTrajIntegratorRK54_h_
#define Kassiopeia_KSTrajIntegratorRK54_h_

#include "KSComponentTemplate.h"
#include "KSMathRK54.h"

#include "KSTrajExactTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajMagneticTypes.h"

namespace Kassiopeia
{

    class KSTrajIntegratorRK54 :
        public KSComponentTemplate< KSTrajIntegratorRK54 >,
        public KSMathRK54< KSTrajExactSystem >,
        public KSMathRK54< KSTrajAdiabaticSystem >,
        public KSMathRK54< KSTrajMagneticSystem >
    {
        public:
            KSTrajIntegratorRK54();
            KSTrajIntegratorRK54( const KSTrajIntegratorRK54& aCopy );
            KSTrajIntegratorRK54* Clone() const;
            virtual ~KSTrajIntegratorRK54();
    };

}

#endif
