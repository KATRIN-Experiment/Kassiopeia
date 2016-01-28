#ifndef Kassiopeia_KSGenValueAngleSpherical_h_
#define Kassiopeia_KSGenValueAngleSpherical_h_

#include "KSGenValue.h"

#include "KField.h"

namespace Kassiopeia
{
    class KSGenValueAngleSpherical :
        public KSComponentTemplate< KSGenValueAngleSpherical, KSGenValue >
    {
        public:
    		KSGenValueAngleSpherical();
    		KSGenValueAngleSpherical( const KSGenValueAngleSpherical& aCopy );
    		KSGenValueAngleSpherical* Clone() const;
            virtual ~KSGenValueAngleSpherical();

        public:
            virtual void DiceValue( vector< double >& aDicedValues );

        public:
            K_SET_GET( double, AngleMin )
            K_SET_GET( double, AngleMax )

    };

}

#endif
