/*
 * KRampedElectric2FieldBuilder.cc
 *
 *  Created on: 16 Jun 2016
 *      Author: wolfgang
 */

#include "KRampedElectric2FieldBuilder.hh"

#include "KEMToolboxBuilder.hh"

using namespace KEMField;
using namespace std;

namespace katrin
{

template<> KRampedElectric2FieldBuilder::~KComplexElement() = default;

STATICINT sKRampedElectric2FieldStructure = KRampedElectric2FieldBuilder::Attribute<std::string>("name") +
                                            KRampedElectric2FieldBuilder::Attribute<std::string>("root_field_1") +
                                            KRampedElectric2FieldBuilder::Attribute<std::string>("root_field_2") +
                                            KRampedElectric2FieldBuilder::Attribute<std::string>("ramping_type") +
                                            KRampedElectric2FieldBuilder::Attribute<int>("num_cycles") +
                                            KRampedElectric2FieldBuilder::Attribute<double>("ramp_up_delay") +
                                            KRampedElectric2FieldBuilder::Attribute<double>("ramp_down_delay") +
                                            KRampedElectric2FieldBuilder::Attribute<double>("ramp_up_time") +
                                            KRampedElectric2FieldBuilder::Attribute<double>("ramp_down_time") +
                                            KRampedElectric2FieldBuilder::Attribute<double>("time_constant") +
                                            KRampedElectric2FieldBuilder::Attribute<double>("time_scaling") +
                                            KRampedElectric2FieldBuilder::Attribute<double>("focus_time") +
                                            KRampedElectric2FieldBuilder::Attribute<double>("focus_exponent") +
                                            KRampedElectric2FieldBuilder::Attribute<double>("potential_scaling") +
                                            KRampedElectric2FieldBuilder::Attribute<bool>("small_spectrometer");


STATICINT sKRampedElectric2Field =
    KEMToolboxBuilder::ComplexElement<KRampedElectric2Field>("ramped_transitional_electric_field");


} /* namespace katrin */
