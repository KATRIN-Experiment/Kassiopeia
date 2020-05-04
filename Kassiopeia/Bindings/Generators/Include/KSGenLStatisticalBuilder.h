//
// Created by Nikolaus Trost on 08.05.15.
//

#ifndef KASPER_KSGENLSTATISTICALBUILDER_H
#define KASPER_KSGENLSTATISTICALBUILDER_H

#include "KComplexElement.hh"
#include "KSGenLStatistical.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSGenLStatistical> KSGenLStatisticalBuilder;

template<> inline bool KSGenLStatisticalBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    return false;
}
}  // namespace katrin

#endif  //KASPER_KSGENLSTATISTICALBUILDER_H
