#include "KSFieldMagneticRampedBuilder.h"
#include "KSRootBuilder.h"

#include "KSMagneticField.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSFieldMagneticRampedBuilder::~KComplexElement()
    {
    }

    STATICINT sKSFieldMagneticRampedStructure =
        KSFieldMagneticRampedBuilder::Attribute< string >( "name" ) +
        KSFieldMagneticRampedBuilder::Attribute< string >( "root_field" ) +
        KSFieldMagneticRampedBuilder::Attribute< string >( "ramping_type" ) +
        KSFieldMagneticRampedBuilder::Attribute< int >( "num_cycles" ) +
        KSFieldMagneticRampedBuilder::Attribute< double >( "ramp_up_delay" ) +
        KSFieldMagneticRampedBuilder::Attribute< double >( "ramp_down_delay" ) +
        KSFieldMagneticRampedBuilder::Attribute< double >( "ramp_up_time" ) +
        KSFieldMagneticRampedBuilder::Attribute< double >( "ramp_down_time" ) +
        KSFieldMagneticRampedBuilder::Attribute< double >( "time_constant" ) +
        KSFieldMagneticRampedBuilder::Attribute< double >( "time_constant_2" ) +
        KSFieldMagneticRampedBuilder::Attribute< double >( "time_scaling" );

    STATICINT sKSFieldMagneticRamped =
        KSRootBuilder::ComplexElement< KSFieldMagneticRamped >( "ksfield_magnetic_ramped" );

    /////////////////////////////////////////////////////////////////////////

    template< >
    KSFieldElectricInducedAzimuthalBuilder::~KComplexElement()
    {
    }

    STATICINT sKSFieldElectricInducedAzimuthalStructure =
        KSFieldElectricInducedAzimuthalBuilder::Attribute< string >( "name" ) +
        KSFieldElectricInducedAzimuthalBuilder::Attribute< string >( "root_field" );

    STATICINT sKSFieldElectricInducedAzimuthal =
        KSRootBuilder::ComplexElement< KSFieldElectricInducedAzimuthal >( "ksfield_electric_induced_azi" );

}
