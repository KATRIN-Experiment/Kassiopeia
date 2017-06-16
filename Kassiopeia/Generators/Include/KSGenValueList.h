#ifndef Kassiopeia_KSGenValueList_h_
#define Kassiopeia_KSGenValueList_h_

#include "KSGenValue.h"

#include "KField.h"

namespace Kassiopeia
{

    class KSGenValueList :
        public KSComponentTemplate< KSGenValueList, KSGenValue >
    {
        public:
    		KSGenValueList();
    		KSGenValueList( const KSGenValueList& aCopy );
    		KSGenValueList* Clone() const;
            virtual ~KSGenValueList();

        public:
            void DiceValue( std::vector< double >& aDicedValues );
            void AddValue( double aValue );

        public:
            std::vector<double> fValues;
    };

}

#endif
