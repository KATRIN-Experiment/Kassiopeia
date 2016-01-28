#ifndef Kassiopeia_KSGenValueRadiusSpherical_h_
#define Kassiopeia_KSGenValueRadiusSpherical_h_

#include "KSGenValue.h"

#include "KField.h"

namespace Kassiopeia
{
    class KSGenValueRadiusSpherical :
        public KSComponentTemplate< KSGenValueRadiusSpherical, KSGenValue >
    {
        public:
            KSGenValueRadiusSpherical();
            KSGenValueRadiusSpherical( const KSGenValueRadiusSpherical& aCopy );
            KSGenValueRadiusSpherical* Clone() const;
            virtual ~KSGenValueRadiusSpherical();

        public:
            virtual void DiceValue( vector< double >& aDicedValues );

        public:
            K_SET_GET( double, RadiusMin )
            K_SET_GET( double, RadiusMax )

    };

}

#endif
