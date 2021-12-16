//
// Created by trost on 12.07.16.
//
#include "KRoot.h"

#include "KElementProcessor.hh"

namespace katrin
{
template<> KRootBuilder::~KComplexElement() = default;

STATICINT sKRoot = KElementProcessor::ComplexElement<KToolbox>("kasper");

STATICINT sEvil = KRootBuilder::ComplexElement<KToolbox>("kasper");
}  // namespace katrin