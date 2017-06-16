#ifndef Kassiopeia_KSGenValueZFrustrum_h_
#define Kassiopeia_KSGenValueZFrustrum_h_

#include "KSGenValue.h"

#include "KField.h"

namespace Kassiopeia
{
    class KSGenValueZFrustrum :
        public KSComponentTemplate< KSGenValueZFrustrum, KSGenValue >
    {
        public:
            KSGenValueZFrustrum();
            KSGenValueZFrustrum( const KSGenValueZFrustrum& aCopy );
            KSGenValueZFrustrum* Clone() const;
            virtual ~KSGenValueZFrustrum();

        public:
            virtual void DiceValue( std::vector< double >& aDicedValues );

        public:
            K_SET_GET( double, r1 )
            K_SET_GET( double, r2 )
            K_SET_GET( double, z1 )
            K_SET_GET( double, z2 )

    };

}

#endif
