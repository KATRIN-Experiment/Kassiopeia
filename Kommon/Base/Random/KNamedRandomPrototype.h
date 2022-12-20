/**
 * @file KNamedRandomPrototype.h
 *
 * @date 01.02.2019
 * @author Valérian Sibille <vsibille@mit.edu>
 *
 */

#ifndef K_NAMED_RANDOM_PROTOTYPE_H
#define K_NAMED_RANDOM_PROTOTYPE_H

#include "KNamed.h"
#include "KRandomPrototype.h"

namespace katrin
{

namespace Kommon
{

template<class XEngineType> class NamedRandomPrototype : public RandomPrototype<XEngineType>, public KNamed
{

    using RandomPrototype<XEngineType>::RandomPrototype;
    void Print(std::ostream& output) const override;
};

using NamedRandomGenerator = NamedRandomPrototype<std::mt19937>;

template<class XEngineType> void NamedRandomPrototype<XEngineType>::Print(std::ostream& output) const
{

    output << "Name: " << this->GetName() << " Seed: " << this->GetSeed();
}

} /*namespace Kommon */

} /* namespace katrin */

#endif
