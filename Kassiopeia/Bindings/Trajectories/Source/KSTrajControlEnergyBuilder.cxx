#include "KSTrajControlEnergyBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajControlEnergyBuilder::~KComplexElement()
    {
    }

    static int sKSTrajControlEnergyStructure =
        KSTrajControlEnergyBuilder::Attribute< string >( "name" ) +
        KSTrajControlEnergyBuilder::Attribute< double >( "lower_limit" ) +
        KSTrajControlEnergyBuilder::Attribute< double >( "upper_limit" );

    static int sToolboxKSTrajControlEnergy =
        KSRootBuilder::ComplexElement< KSTrajControlEnergy >( "kstraj_control_energy" );

}
