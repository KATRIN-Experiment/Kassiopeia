#include "KSTrajControlMomentumNumericalErrorBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajControlMomentumNumericalErrorBuilder::~KComplexElement() = default;

STATICINT sKSTrajControlMomentumNumericalErrorStructure =
    KSTrajControlMomentumNumericalErrorBuilder::Attribute<std::string>("name") +
    KSTrajControlMomentumNumericalErrorBuilder::Attribute<double>("absolute_momentum_error") +
    KSTrajControlMomentumNumericalErrorBuilder::Attribute<double>("safety_factor") +
    KSTrajControlMomentumNumericalErrorBuilder::Attribute<double>("solver_order");

STATICINT sToolboxKSTrajControlMomentumNumericalError =
    KSRootBuilder::ComplexElement<KSTrajControlMomentumNumericalError>("kstraj_control_momentum_numerical_error");

}  // namespace katrin
