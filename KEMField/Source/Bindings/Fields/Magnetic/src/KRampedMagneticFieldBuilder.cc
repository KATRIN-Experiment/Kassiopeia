/*
 * KRampedMagneticFieldBuilder.cc
 *
 *  Created on: 8 Apr 2016
 *      Author: wolfgang
 */

#include "KRampedMagneticFieldBuilder.hh"

#include "KEMToolboxBuilder.hh"

using namespace KEMField;
using namespace std;

namespace katrin
{

template<> KRampedMagneticFieldBuilder::~KComplexElement() = default;

STATICINT sKRampedMagneticFieldStructure = KRampedMagneticFieldBuilder::Attribute<std::string>("name") +
                                           KRampedMagneticFieldBuilder::Attribute<std::string>("root_field") +
                                           KRampedMagneticFieldBuilder::Attribute<std::string>("ramping_type") +
                                           KRampedMagneticFieldBuilder::Attribute<int>("num_cycles") +
                                           KRampedMagneticFieldBuilder::Attribute<double>("ramp_up_delay") +
                                           KRampedMagneticFieldBuilder::Attribute<double>("ramp_down_delay") +
                                           KRampedMagneticFieldBuilder::Attribute<double>("ramp_up_time") +
                                           KRampedMagneticFieldBuilder::Attribute<double>("ramp_down_time") +
                                           KRampedMagneticFieldBuilder::Attribute<double>("time_constant") +
                                           KRampedMagneticFieldBuilder::Attribute<double>("time_constant_2") +
                                           KRampedMagneticFieldBuilder::Attribute<double>("time_scaling");

STATICINT sKRampedMagneticField = KEMToolboxBuilder::ComplexElement<KRampedMagneticField>("ramped_magnetic_field");

} /* namespace katrin */
