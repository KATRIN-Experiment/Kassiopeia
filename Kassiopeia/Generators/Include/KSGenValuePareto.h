#ifndef Kassiopeia_KSGenValuePareto_h_
#define Kassiopeia_KSGenValuePareto_h_

#include "KSGenValue.h"

#include "KField.h"
#include "KMathBracketingSolver.h"
using katrin::KMathBracketingSolver;

namespace Kassiopeia
{
    class KSGenValuePareto :
        public KSComponentTemplate< KSGenValuePareto, KSGenValue >
    {
        public:
            KSGenValuePareto();
            KSGenValuePareto( const KSGenValuePareto& aCopy );
            KSGenValuePareto* Clone() const;
            virtual ~KSGenValuePareto();

        public:
            virtual void DiceValue( vector< double >& aDicedValues );

        public:
            K_SET_GET( double, Slope )
            K_SET_GET( double, Cutoff )
            K_SET_GET( double, Offset )
            K_SET_GET( double, ValueMin )
            K_SET_GET( double, ValueMax )

        public:
            void InitializeComponent();
            void DeinitializeComponent();

        protected:
            double fValueParetoMin;
            double fValueParetoMax;
    };

}

#endif
