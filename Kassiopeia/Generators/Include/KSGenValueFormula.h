#ifndef Kassiopeia_KSGenValueFormula_h_
#define Kassiopeia_KSGenValueFormula_h_

#include "KSGenValue.h"

#include "KField.h"

#include "TF1.h"

namespace Kassiopeia
{
    class KSGenValueFormula :
        public KSComponentTemplate< KSGenValueFormula, KSGenValue >
    {
        public:
            KSGenValueFormula();
            KSGenValueFormula( const KSGenValueFormula& aCopy );
            KSGenValueFormula* Clone() const;
            virtual ~KSGenValueFormula();

        public:
            virtual void DiceValue( std::vector< double >& aDicedValues );

        public:
            ;K_SET_GET( double, ValueMin );
            ;K_SET_GET( double, ValueMax );
            ;K_SET_GET( std::string, ValueFormula );

        public:
            void InitializeComponent();
            void DeinitializeComponent();

        protected:
            TF1* fValueFunction;
    };

}

#endif
