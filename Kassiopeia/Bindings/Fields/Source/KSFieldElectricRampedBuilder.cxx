#include "KSFieldElectricRampedBuilder.h"
#include "KSRootBuilder.h"

#include "KSElectricField.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSFieldElectricRampedBuilder::~KComplexElement()
    {
    }

    static int sKSFieldElectricRampedStructure =
        KSFieldElectricRampedBuilder::Attribute< string >( "name" ) +
        KSFieldElectricRampedBuilder::Attribute< string >( "root_field" ) +
        KSFieldElectricRampedBuilder::Attribute< string >( "ramping_type" ) +
        KSFieldElectricRampedBuilder::Attribute< int >( "num_cycles" ) +
        KSFieldElectricRampedBuilder::Attribute< double >( "ramp_up_delay" ) +
        KSFieldElectricRampedBuilder::Attribute< double >( "ramp_down_delay" ) +
        KSFieldElectricRampedBuilder::Attribute< double >( "ramp_up_time" ) +
        KSFieldElectricRampedBuilder::Attribute< double >( "ramp_down_time" ) +
        KSFieldElectricRampedBuilder::Attribute< double >( "time_constant" ) +
        KSFieldElectricRampedBuilder::Attribute< double >( "time_scaling" );

    static int sKSFieldElectricRamped =
        KSRootBuilder::ComplexElement< KSFieldElectricRamped >( "ksfield_electric_ramped" );

}
