#ifndef Kassiopeia_KSTrajIntegratorRK54_h_
#define Kassiopeia_KSTrajIntegratorRK54_h_

#include "KSComponentTemplate.h"
#include "KSMathRKF54.h"

#include "KSTrajExactTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajElectricTypes.h"
#include "KSTrajMagneticTypes.h"

namespace Kassiopeia
{

    class KSTrajIntegratorRK54 :
        public KSComponentTemplate< KSTrajIntegratorRK54 >,
        public KSMathRKF54< KSTrajExactSystem >,
        public KSMathRKF54< KSTrajAdiabaticSystem >,
        public KSMathRKF54< KSTrajElectricSystem >,
        public KSMathRKF54< KSTrajMagneticSystem >
    {
        public:
            KSTrajIntegratorRK54();
            KSTrajIntegratorRK54( const KSTrajIntegratorRK54& aCopy );
            KSTrajIntegratorRK54* Clone() const;
            virtual ~KSTrajIntegratorRK54();
    };

}

#endif
