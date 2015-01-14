#include "KSTermMinEnergyBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTermMinEnergyBuilder::~KComplexElement()
    {
    }

    static int sKSTermMinEnergyStructure =
        KSTermMinEnergyBuilder::Attribute< string >( "name" ) +
        KSTermMinEnergyBuilder::Attribute< double >( "energy" );

    static int sKSTermMinEnergy =
        KSRootBuilder::ComplexElement< KSTermMinEnergy >( "ksterm_min_energy" );

}
