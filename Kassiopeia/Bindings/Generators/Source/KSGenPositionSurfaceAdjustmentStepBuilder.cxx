/*
 * KSGenPositionSurfaceAdjustmentStep.h
 *
 *  Created on: 28.01.2015
 *      Author: Nikolaus Trost
 */

#include "KSGenPositionSurfaceAdjustmentStepBuilder.h"


using namespace Kassiopeia;
using namespace std;

namespace katrin
{
template<> KSGenPositionSurfaceAdjustmentStepBuilder::~KComplexElement() {}

STATICINT sKSKSGenPositionSurfaceAdjustmentStepStructure =
    KSGenPositionSurfaceAdjustmentStepBuilder::Attribute<string>("name") +
    KSGenPositionSurfaceAdjustmentStepBuilder::Attribute<double>("length");

STATICINT sKSGenPositionSurfaceAdjustmentStep =
    KSRootBuilder::ComplexElement<KSGenPositionSurfaceAdjustmentStep>("ksgen_position_surface_adjustment_step");

}  // namespace katrin
