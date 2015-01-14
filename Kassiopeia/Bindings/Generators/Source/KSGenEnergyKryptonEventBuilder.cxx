#include "KSGenEnergyKryptonEventBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSGenEnergyKryptonEventBuilder::~KComplexElement()
    {
    }

    static int sKSGenEnergyKryptonEventStructure =
        KSGenEnergyKryptonEventBuilder::Attribute< string >( "name" ) +
        KSGenEnergyKryptonEventBuilder::Attribute< bool >( "force_conversion" ) +
        KSGenEnergyKryptonEventBuilder::Attribute< bool >( "do_conversion" ) +
        KSGenEnergyKryptonEventBuilder::Attribute< bool >( "do_auger" );


    static int sKSGenEnergyKryptonEvent =
        KSRootBuilder::ComplexElement< KSGenEnergyKryptonEvent >( "ksgen_energy_krypton_event" );

}
