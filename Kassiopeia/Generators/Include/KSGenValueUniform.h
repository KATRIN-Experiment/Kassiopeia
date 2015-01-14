#ifndef Kassiopeia_KSGenValueUniform_h_
#define Kassiopeia_KSGenValueUniform_h_

#include "KSGenValue.h"

#include "KField.h"

namespace Kassiopeia
{

    class KSGenValueUniform :
        public KSComponentTemplate< KSGenValueUniform, KSGenValue >
    {
        public:
            KSGenValueUniform();
            KSGenValueUniform( const KSGenValueUniform& aCopy );
            KSGenValueUniform* Clone() const;
            virtual ~KSGenValueUniform();

        public:
            virtual void DiceValue( vector< double >& aDicedValues );

        public:
            ;K_SET_GET( double, ValueMin );
            ;K_SET_GET( double, ValueMax );
    };

}

#endif
