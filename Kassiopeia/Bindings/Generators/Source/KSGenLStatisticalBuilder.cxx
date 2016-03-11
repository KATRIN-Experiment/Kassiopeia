//
// Created by Nikolaus Trost on 08.05.15.
//

#include "KSGenLStatisticalBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin {

    template<>
    KSGenLStatisticalBuilder::~KComplexElement() {
    }

    STATICINT sKSGenLStatisticalStructure =
            KSGenLStatisticalBuilder::Attribute<string>("name");

    STATICINT sKSGenLStatistical =
            KSRootBuilder::ComplexElement<KSGenLStatistical>("ksgen_l_statistical");
}
