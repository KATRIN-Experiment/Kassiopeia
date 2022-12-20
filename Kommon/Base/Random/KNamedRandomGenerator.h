/**
 * @file KNamedRandomGenerator.h
 *
 * @date 01.02.2019
 * @author Valérian Sibille <vsibille@mit.edu>
 *
 */

#ifndef K_NAMED_RANDOM_GENERATOR_H
#define K_NAMED_RANDOM_GENERATOR_H

#include "KNamedRandomPrototype.h"

namespace katrin
{

namespace Kommon
{

using NamedRandomGenerator = NamedRandomPrototype<std::mt19937>;

} /*namespace Kommon */

} /* namespace katrin */

#endif
