#ifndef Kassiopeia_KSGenValueFix_h_
#define Kassiopeia_KSGenValueFix_h_

#include "KSGenValue.h"

#include "KField.h"

namespace Kassiopeia
{

    class KSGenValueFix :
        public KSComponentTemplate< KSGenValueFix, KSGenValue >
    {
        public:
            KSGenValueFix();
            KSGenValueFix( const KSGenValueFix& aCopy );
            KSGenValueFix* Clone() const;
            virtual ~KSGenValueFix();

        public:
            virtual void DiceValue( vector< double >& aDicedValues );

        public:
           K_SET_GET( double, Value )
    };

}

#endif
