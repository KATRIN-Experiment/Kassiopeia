#ifndef Kassiopeia_KSGenValueGeneralizedGauss_h_
#define Kassiopeia_KSGenValueGeneralizedGauss_h_

#include "KSGenValue.h"

#include "KField.h"
#include "KMathBracketingSolver.h"
using katrin::KMathBracketingSolver;

namespace Kassiopeia
{
    class KSGenValueGeneralizedGauss :
        public KSComponentTemplate< KSGenValueGeneralizedGauss, KSGenValue >
    {
        public:
            KSGenValueGeneralizedGauss();
            KSGenValueGeneralizedGauss( const KSGenValueGeneralizedGauss& aCopy );
            KSGenValueGeneralizedGauss* Clone() const;
            virtual ~KSGenValueGeneralizedGauss();

        public:
            virtual void DiceValue( std::vector< double >& aDicedValues );

        public:
            K_SET_GET( double, ValueMin )
            K_SET_GET( double, ValueMax )
            K_SET_GET( double, ValueMean )
            K_SET_GET( double, ValueSigma )
            K_SET_GET( double, ValueSkew )

        protected:
            double ValueFunction( const double& aValue ) const;
            KMathBracketingSolver fSolver;
    };

}

#endif
