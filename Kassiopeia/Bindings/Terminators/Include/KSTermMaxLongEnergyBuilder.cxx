#include "KSTermMaxLongEnergyBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTermMaxLongEnergyBuilder::~KComplexElement()
    {
    }

    static int sKSTermMaxLongEnergyStructure =
            KSTermMaxLongEnergyBuilder::Attribute< string >( "name" ) +
            KSTermMaxLongEnergyBuilder::Attribute< double >( "long_energy" );

    static int sKSTermMaxLongEnergy =
            KSRootBuilder::ComplexElement< KSTermMaxLongEnergy >( "ksterm_max_long_energy" );

}
