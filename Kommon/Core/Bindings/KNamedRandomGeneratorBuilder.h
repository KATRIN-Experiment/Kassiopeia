/*!
 * @file KNamedRandomGeneratorBuilder.h
 * @author Valérian Sibille <vsibille@mit.edu>
 */
#ifndef NAMED_RANDOM_GENERATOR_BUILDER_H
#define NAMED_RANDOM_GENERATOR_BUILDER_H

#include "KComplexElement.hh"
#include "KNamedRandomGenerator.h"
#include "KToolbox.h"

namespace katrin
{

using NamedRandomGeneratorBuilder = KComplexElement<Kommon::NamedRandomGenerator>;

template<> bool NamedRandomGeneratorBuilder::AddAttribute(KContainer* aContainer);

template<> bool NamedRandomGeneratorBuilder::AddElement(KContainer* aContainer);

}  // namespace katrin

#endif
