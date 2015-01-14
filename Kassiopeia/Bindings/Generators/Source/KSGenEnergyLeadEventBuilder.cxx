#include "KSGenEnergyLeadEventBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSGenEnergyLeadEventBuilder::~KComplexElement()
    {
    }

    static int sKSGenEnergyLeadEventStructure =
        KSGenEnergyLeadEventBuilder::Attribute< string >( "name" ) +
        KSGenEnergyLeadEventBuilder::Attribute< bool >( "force_conversion" ) +
        KSGenEnergyLeadEventBuilder::Attribute< bool >( "do_conversion" ) +
        KSGenEnergyLeadEventBuilder::Attribute< bool >( "do_auger" );

    static int sKSGenEnergyLeadEvent =
        KSRootBuilder::ComplexElement< KSGenEnergyLeadEvent >( "ksgen_energy_lead_event" );

}
