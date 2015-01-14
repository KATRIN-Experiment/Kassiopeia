#include "KSGenValueFormulaBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSGenValueFormulaBuilder::~KComplexElement()
    {
    }

    static int sKSGenValueFormulaStructure =
        KSGenValueFormulaBuilder::Attribute< string >( "name" ) +
        KSGenValueFormulaBuilder::Attribute< double >( "value_min" ) +
        KSGenValueFormulaBuilder::Attribute< double >( "value_max" ) +
        KSGenValueFormulaBuilder::Attribute< string >( "value_formula" );

    static int sKSGenValueFormula =
        KSRootBuilder::ComplexElement< KSGenValueFormula >( "ksgen_value_formula" );

}
