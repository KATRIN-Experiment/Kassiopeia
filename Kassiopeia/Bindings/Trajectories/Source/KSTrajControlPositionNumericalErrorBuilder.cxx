#include "KSTrajControlPositionNumericalErrorBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajControlPositionNumericalErrorBuilder::~KComplexElement() = default;

STATICINT sKSTrajControlPositionNumericalErrorStructure =
    KSTrajControlPositionNumericalErrorBuilder::Attribute<std::string>("name") +
    KSTrajControlPositionNumericalErrorBuilder::Attribute<double>("absolute_position_error") +
    KSTrajControlPositionNumericalErrorBuilder::Attribute<double>("safety_factor") +
    KSTrajControlPositionNumericalErrorBuilder::Attribute<double>("solver_order");

STATICINT sToolboxKSTrajControlPositionNumericalError =
    KSRootBuilder::ComplexElement<KSTrajControlPositionNumericalError>("kstraj_control_position_numerical_error");

}  // namespace katrin
