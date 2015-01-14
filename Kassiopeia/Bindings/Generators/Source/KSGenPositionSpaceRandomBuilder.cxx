/*
 * KSGenPositionSpaceBuilder.cxx
 *
 *  Created on: 01.07.2014
 *      Author: oertlin
 */

#include "KSGenPositionSpaceRandomBuilder.h"

using namespace Kassiopeia;
namespace katrin
{
    template<>
    KSGenPositionSpaceRandomBuilder::~KComplexElement() {}

    static int sKSGenPositionSpaceRandomStructure =
		KSGenPositionSpaceRandomBuilder::Attribute<string>("name") +
        KSGenPositionSpaceRandomBuilder::Attribute<string>("spaces");

    static int sKSGenPositionSpaceRandom =
		KSRootBuilder::ComplexElement<KSGenPositionSpaceRandom>("ksgen_position_space_random");

}
