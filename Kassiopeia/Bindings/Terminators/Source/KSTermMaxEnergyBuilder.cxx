#include "KSTermMaxEnergyBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTermMaxEnergyBuilder::~KComplexElement()
    {
    }

    static int sKSTermMaxEnergyStructure =
        KSTermMaxEnergyBuilder::Attribute< string >( "name" ) +
        KSTermMaxEnergyBuilder::Attribute< double >( "energy" );

    static int sKSTermMaxEnergy =
        KSRootBuilder::ComplexElement< KSTermMaxEnergy >( "ksterm_max_energy" );

}
