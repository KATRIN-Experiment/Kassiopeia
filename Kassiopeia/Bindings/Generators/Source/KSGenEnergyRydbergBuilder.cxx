//
// Created by trost on 14.03.16.
//

#include "KSGenEnergyRydbergBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template< >
KSGenEnergyRydbergBuilder::~KComplexElement()
{
}

STATICINT sKSGenEnergyRydbergStructure =
    KSGenEnergyRydbergBuilder::Attribute< string >( "name" ) +
    KSGenEnergyRydbergBuilder::Attribute< double >( "ionization_energy" ) +
    KSGenEnergyRydbergBuilder::Attribute< double >( "deposited_energy" );

STATICINT sKSGenEnergyRydberg =
    KSRootBuilder::ComplexElement< KSGenEnergyRydberg >( "ksgen_energy_rydberg" );

}