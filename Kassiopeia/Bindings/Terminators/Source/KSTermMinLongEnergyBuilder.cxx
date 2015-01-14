#include "KSTermMinLongEnergyBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTermMinLongEnergyBuilder::~KComplexElement()
    {
    }

    static int sKSTermMinLongEnergyStructure =
            KSTermMinLongEnergyBuilder::Attribute< string >( "name" ) +
            KSTermMinLongEnergyBuilder::Attribute< double >( "long_energy" );

    static int sKSTermMinLongEnergy =
            KSRootBuilder::ComplexElement< KSTermMinLongEnergy >( "ksterm_min_long_energy" );

}
