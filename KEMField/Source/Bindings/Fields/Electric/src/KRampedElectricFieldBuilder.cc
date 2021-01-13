/*
 * KRampedElectricFieldBuilder.cc
 *
 *  Created on: 31 May 2016
 *      Author: wolfgang
 */

#include "KRampedElectricFieldBuilder.hh"

#include "KEMToolboxBuilder.hh"

using namespace KEMField;
using namespace std;

namespace katrin
{

template<> KRampedElectricFieldBuilder::~KComplexElement() = default;

STATICINT sKRampedElectricFieldStructure = KRampedElectricFieldBuilder::Attribute<std::string>("name") +
                                           KRampedElectricFieldBuilder::Attribute<std::string>("root_field") +
                                           KRampedElectricFieldBuilder::Attribute<std::string>("ramping_type") +
                                           KRampedElectricFieldBuilder::Attribute<int>("num_cycles") +
                                           KRampedElectricFieldBuilder::Attribute<double>("ramp_up_delay") +
                                           KRampedElectricFieldBuilder::Attribute<double>("ramp_down_delay") +
                                           KRampedElectricFieldBuilder::Attribute<double>("ramp_up_time") +
                                           KRampedElectricFieldBuilder::Attribute<double>("ramp_down_time") +
                                           KRampedElectricFieldBuilder::Attribute<double>("time_constant") +
                                           KRampedElectricFieldBuilder::Attribute<double>("time_scaling");

STATICINT sKRampedElectricField = KEMToolboxBuilder::ComplexElement<KRampedElectricField>("ramped_electric_field");

} /* namespace katrin */
