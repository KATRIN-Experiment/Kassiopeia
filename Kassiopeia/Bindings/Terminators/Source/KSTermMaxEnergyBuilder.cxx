#include "KSTermMaxEnergyBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTermMaxEnergyBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTermMaxEnergyStructure =
        KSTermMaxEnergyBuilder::Attribute< string >( "name" ) +
        KSTermMaxEnergyBuilder::Attribute< double >( "energy" );

    STATICINT sKSTermMaxEnergy =
        KSRootBuilder::ComplexElement< KSTermMaxEnergy >( "ksterm_max_energy" );

}
