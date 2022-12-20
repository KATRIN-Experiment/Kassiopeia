/**
 * @file KRandom.h
 *
 * @date 24.11.2013
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 *
 */

#ifndef KRANDOM_H_
#define KRANDOM_H_

#include "KNonCopyable.h"
#include "KRandomPrototype.h"
#include "KSingleton.h"

namespace katrin
{

namespace Kommon
{

template<class XEngineType>
class RandomSingleton :
    public RandomPrototype<XEngineType>,
    public KSingleton<RandomSingleton<XEngineType>>,
    KNonCopyable
{

    using RandomPrototype<XEngineType>::RandomPrototype;
};

} /*namespace Kommon */

using KRandom = Kommon::RandomSingleton<std::mt19937>;

} /* namespace katrin */

#endif /* KRANDOM_H_ */
