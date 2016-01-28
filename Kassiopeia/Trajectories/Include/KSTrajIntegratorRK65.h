#ifndef Kassiopeia_KSTrajIntegratorRK65_h_
#define Kassiopeia_KSTrajIntegratorRK65_h_

#include "KSComponentTemplate.h"
#include "KSMathRK65.h"

#include "KSTrajExactTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajElectricTypes.h"
#include "KSTrajMagneticTypes.h"

namespace Kassiopeia
{

    class KSTrajIntegratorRK65 :
        public KSComponentTemplate< KSTrajIntegratorRK65 >,
        public KSMathRK65< KSTrajExactSystem >,
        public KSMathRK65< KSTrajAdiabaticSystem >,
        public KSMathRK65< KSTrajElectricSystem >,
        public KSMathRK65< KSTrajMagneticSystem >
    {
        public:
            KSTrajIntegratorRK65();
            KSTrajIntegratorRK65( const KSTrajIntegratorRK65& aCopy );
            KSTrajIntegratorRK65* Clone() const;
            virtual ~KSTrajIntegratorRK65();
    };

}

#endif
