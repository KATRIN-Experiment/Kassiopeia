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

STATICINT sKRampedMagneticFieldStructure =
    //! the name of the magnetic field
    KRampedMagneticFieldBuilder::Attribute<std::string>("name") +
    //! the name of the magnetic field that is to be varied with time
    KRampedMagneticFieldBuilder::Attribute<std::string>("root_field") +
    //! linear, exponential inversion and flipbox are the options
    KRampedMagneticFieldBuilder::Attribute<std::string>("ramping_type") +
    //! how often the ramping up and down cycle is repeated
    KRampedMagneticFieldBuilder::Attribute<int>("num_cycles") +
    //! delay before ramping up
    KRampedMagneticFieldBuilder::Attribute<double>("ramp_up_delay") +
    //! delay before ramping down af
    KRampedMagneticFieldBuilder::Attribute<double>("ramp_down_delay") +
    //! time the linear ramp up takes
    KRampedMagneticFieldBuilder::Attribute<double>("ramp_up_time") +
    //! time the linear ramp down takes
    KRampedMagneticFieldBuilder::Attribute<double>("ramp_down_time") +
    //! time constant for exponential ramping, otherwise unused value
    KRampedMagneticFieldBuilder::Attribute<double>("time_constant") +
    KRampedMagneticFieldBuilder::Attribute<double>("time_constant_2") +
    //! scale all times by a factor for quick frequency change
    KRampedMagneticFieldBuilder::Attribute<double>("time_scaling");

STATICINT sKRampedMagneticField = KEMToolboxBuilder::ComplexElement<KRampedMagneticField>("ramped_magnetic_field");

} /* namespace katrin */
