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

STATICINT sKRampedElectricFieldStructure =
    //! the name of the electric field
    KRampedElectricFieldBuilder::Attribute<std::string>("name") +
    //! the name of the electric field that is to be varied with time
    KRampedElectricFieldBuilder::Attribute<std::string>("root_field") +
    //! linear, exponential and sinus are the options
    KRampedElectricFieldBuilder::Attribute<std::string>("ramping_type") +
    //! how often the ramping up and down cycle is repeated
    KRampedElectricFieldBuilder::Attribute<int>("num_cycles") +
    //! delay before ramping up
    KRampedElectricFieldBuilder::Attribute<double>("ramp_up_delay") +
    //! delay before ramping down af
    KRampedElectricFieldBuilder::Attribute<double>("ramp_down_delay") +
    //! time the linear ramp up takes
    KRampedElectricFieldBuilder::Attribute<double>("ramp_up_time") +
    //! time the linear ramp down takes
    KRampedElectricFieldBuilder::Attribute<double>("ramp_down_time") +
    //! time constant for exponential ramping, otherwise unused value
    KRampedElectricFieldBuilder::Attribute<double>("time_constant") +
    //! scale all times by a factor for quick frequency change
    KRampedElectricFieldBuilder::Attribute<double>("time_scaling");

STATICINT sKRampedElectricField = KEMToolboxBuilder::ComplexElement<KRampedElectricField>("ramped_electric_field");

} /* namespace katrin */
