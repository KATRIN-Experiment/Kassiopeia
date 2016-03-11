//
// Created by Nikolaus Trost on 07.05.15.
//

#ifndef KASPER_KSGENLUNIFORMMAXNBUILDER_H
#define KASPER_KSGENLUNIFORMMAXNBUILDER_H

#include "KComplexElement.hh"
#include "KSGenLUniformMaxN.h"
#include "KSToolbox.h"

using namespace Kassiopeia;
namespace katrin {

    typedef KComplexElement <KSGenLUniformMaxN> KSGenLUniformMaxNBuilder;

    template<>
    inline bool KSGenLUniformMaxNBuilder::AddAttribute(KContainer *aContainer) {
        if (aContainer->GetName() == "name") {
            aContainer->CopyTo(fObject, &KNamed::SetName);
            return true;
        }
        return false;
    }
}

#endif //KASPER_KSGENLUNIFORMMAXNBUILDER_H
