#ifndef Kassiopeia_KSTrajIntegratorRK86_h_
#define Kassiopeia_KSTrajIntegratorRK86_h_

#include "KSComponentTemplate.h"
#include "KSMathRK86.h"

#include "KSTrajExactTypes.h"
#include "KSTrajExactSpinTypes.h"
#include "KSTrajAdiabaticSpinTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajElectricTypes.h"
#include "KSTrajMagneticTypes.h"

namespace Kassiopeia
{

    class KSTrajIntegratorRK86 :
        public KSComponentTemplate< KSTrajIntegratorRK86 >,
        public KSMathRK86< KSTrajExactSystem >,
        public KSMathRK86< KSTrajExactSpinSystem >,
        public KSMathRK86< KSTrajAdiabaticSpinSystem >,
        public KSMathRK86< KSTrajAdiabaticSystem >,
        public KSMathRK86< KSTrajElectricSystem >,
        public KSMathRK86< KSTrajMagneticSystem >
    {
        public:
            KSTrajIntegratorRK86();
            KSTrajIntegratorRK86( const KSTrajIntegratorRK86& aCopy );
            KSTrajIntegratorRK86* Clone() const;
            virtual ~KSTrajIntegratorRK86();
    };

}

#endif
