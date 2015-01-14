#ifndef Kassiopeia_KSGenValueGauss_h_
#define Kassiopeia_KSGenValueGauss_h_

#include "KSGenValue.h"

#include "KField.h"
#include "KMathBracketingSolver.h"
using katrin::KMathBracketingSolver;

namespace Kassiopeia
{
    class KSGenValueGauss :
        public KSComponentTemplate< KSGenValueGauss, KSGenValue >
    {
        public:
            KSGenValueGauss();
            KSGenValueGauss( const KSGenValueGauss& aCopy );
            KSGenValueGauss* Clone() const;
            virtual ~KSGenValueGauss();

        public:
            virtual void DiceValue( vector< double >& aDicedValues );

        public:
            ;K_SET_GET( double, ValueMin );
            ;K_SET_GET( double, ValueMax );
            ;K_SET_GET( double, ValueMean );
            ;K_SET_GET( double, ValueSigma );

        protected:
            double ValueFunction( const double& aValue ) const;
            KMathBracketingSolver fSolver;
    };

}

#endif
