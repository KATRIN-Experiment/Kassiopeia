#include "KSTrajControlMomentumNumericalErrorBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajControlMomentumNumericalErrorBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTrajControlMomentumNumericalErrorStructure =
        KSTrajControlMomentumNumericalErrorBuilder::Attribute< string >( "name" ) +
        KSTrajControlMomentumNumericalErrorBuilder::Attribute< double >( "absolute_momentum_error" ) +
        KSTrajControlMomentumNumericalErrorBuilder::Attribute< double >( "safety_factor" ) +
        KSTrajControlMomentumNumericalErrorBuilder::Attribute< double >( "solver_order" );

    STATICINT sToolboxKSTrajControlMomentumNumericalError =
        KSRootBuilder::ComplexElement< KSTrajControlMomentumNumericalError >( "kstraj_control_momentum_numerical_error" );

}
