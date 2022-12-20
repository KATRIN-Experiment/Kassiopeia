/*
 * KFRandomBuilder.h
 *
 *  Created on: 22.06.2016
 *      Author: marco.kleesiek@kit.edu
 */

#ifndef KOMMON_BINDINGS_KRANDOMBUILDER_H_
#define KOMMON_BINDINGS_KRANDOMBUILDER_H_

#include "KComplexElement.hh"
#include "KNamed.h"
#include "KRandom.h"


namespace katrin
{
class KDummyRandom : public KNamed
{};

typedef KComplexElement<KDummyRandom> KRandomBuilder;

template<> bool KRandomBuilder::Begin();

template<> bool KRandomBuilder::AddAttribute(KContainer* aContainer);

template<> bool KRandomBuilder::End();

} /* namespace katrin */

#endif /* KAFIT_BINDINGS_KFRANDOMBUILDER_H_ */
