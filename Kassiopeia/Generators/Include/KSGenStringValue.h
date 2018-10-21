#ifndef _Kassiopeia_KSGenStringValue_h_
#define _Kassiopeia_KSGenStringValue_h_

#include "KSComponentTemplate.h"

#include <vector>
using std::vector;

namespace Kassiopeia
{

    class KSGenStringValue :
        public KSComponentTemplate< KSGenStringValue >
    {
        public:
            KSGenStringValue();
            virtual ~KSGenStringValue();

        public:
            virtual void DiceValue( std::vector< std::string >& aDicedValue ) = 0;
    };

}

#endif
