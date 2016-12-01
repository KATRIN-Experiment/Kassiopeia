#include "KSTrajControlSpinPrecessionBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSTrajControlSpinPrecessionBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTrajControlSpinPrecessionStructure =
        KSTrajControlSpinPrecessionBuilder::Attribute< string >( "name" ) +
        KSTrajControlSpinPrecessionBuilder::Attribute< double >( "fraction" );

    STATICINT sToolboxKSTrajControlSpinPrecession =
        KSRootBuilder::ComplexElement< KSTrajControlSpinPrecession >( "kstraj_control_spin_precession" );

}
