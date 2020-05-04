/*
 * KFRandomBuilder.cxx
 *
 *  Created on: 22.06.2016
 *      Author: marco.kleesiek@kit.edu
 */
#include "KRandomBuilder.h"

#include "KElementProcessor.hh"
#include "KRoot.h"

namespace katrin
{

STATICINT sKRandomHook = KRootBuilder::ComplexElement<KDummyRandom>("Random");
STATICINT sKRandomHookCompat = KElementProcessor::ComplexElement<KDummyRandom>("Random");

STATICINT sKRandomStructure = KRandomBuilder::Attribute<int32_t>("Seed");

template<> bool KRandomBuilder::Begin()
{
    return true;
}

template<> bool KRandomBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "Seed") {
        KRandom::GetInstance().SetSeed(std::max<int32_t>(0, aContainer->AsReference<int32_t>()));
        return true;
    }

    return false;
}

template<> bool KRandomBuilder::End()
{
    return true;
}

} /* namespace katrin */
