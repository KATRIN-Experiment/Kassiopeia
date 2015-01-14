#include "KSIntCalculatorConstantBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSIntCalculatorConstantBuilder::~KComplexElement()
    {
    }

    static int sKSIntCalculatorConstantStructure =
        KSIntCalculatorConstantBuilder::Attribute< string >( "name" ) +
        KSIntCalculatorConstantBuilder::Attribute< double >( "cross_section" );

    static int sToolboxKSIntCalculatorConstant =
        KSRootBuilder::ComplexElement< KSIntCalculatorConstant >( "ksint_calculator_constant" );
}
