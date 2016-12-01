//
// Created by Nikolaus Trost on 08.05.15.
//

#include "KSGenLStatisticalBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin {

    template<>
    KSGenLStatisticalBuilder::~KComplexElement() {
    }

    STATICINT sKSGenLStatisticalStructure =
            KSGenLStatisticalBuilder::Attribute<string>("name");

    STATICINT sKSGenLStatistical =
            KSRootBuilder::ComplexElement<KSGenLStatistical>("ksgen_l_statistical");
}
