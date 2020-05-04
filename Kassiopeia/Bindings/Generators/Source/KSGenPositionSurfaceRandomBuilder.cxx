/*
 * KSGenPositionSurfaceRandomBuilder.cxx
 *
 *  Created on: 17.09.2014
 *      Author: J. Behrens
 */

#include "KSGenPositionSurfaceRandomBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{
template<> KSGenPositionSurfaceRandomBuilder::~KComplexElement() {}

STATICINT sKSGenPositionSurfaceRandomStructure = KSGenPositionSurfaceRandomBuilder::Attribute<string>("name") +
                                                 KSGenPositionSurfaceRandomBuilder::Attribute<string>("surfaces");

STATICINT sKSGenPositionSurfaceRandom =
    KSRootBuilder::ComplexElement<KSGenPositionSurfaceRandom>("ksgen_position_surface_random");

}  // namespace katrin
