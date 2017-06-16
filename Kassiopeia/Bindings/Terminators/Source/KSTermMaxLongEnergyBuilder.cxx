#include "KSTermMaxLongEnergyBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSTermMaxLongEnergyBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTermMaxLongEnergyStructure =
            KSTermMaxLongEnergyBuilder::Attribute< string >( "name" ) +
            KSTermMaxLongEnergyBuilder::Attribute< double >( "long_energy" );

    STATICINT sKSTermMaxLongEnergy =
            KSRootBuilder::ComplexElement< KSTermMaxLongEnergy >( "ksterm_max_long_energy" );

}
