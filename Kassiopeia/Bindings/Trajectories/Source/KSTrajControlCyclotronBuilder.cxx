#include "KSTrajControlCyclotronBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajControlCyclotronBuilder::~KComplexElement() = default;

STATICINT sKSTrajControlCyclotronStructure =
    //! the name of this control.
    KSTrajControlCyclotronBuilder::Attribute<std::string>("name") +
    //! the fraction by which to multiply the cyclotron period to obtain the suggested temporal step.
    KSTrajControlCyclotronBuilder::Attribute<double>("fraction");

STATICINT sToolboxKSTrajControlCyclotron =
    KSRootBuilder::ComplexElement<KSTrajControlCyclotron>("kstraj_control_cyclotron");

}  // namespace katrin
