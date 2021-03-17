#include "KSTrajControlCyclotronBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajControlCyclotronBuilder::~KComplexElement() = default;

STATICINT sKSTrajControlCyclotronStructure = KSTrajControlCyclotronBuilder::Attribute<std::string>("name") +
                                             KSTrajControlCyclotronBuilder::Attribute<double>("fraction");

STATICINT sToolboxKSTrajControlCyclotron =
    KSRootBuilder::ComplexElement<KSTrajControlCyclotron>("kstraj_control_cyclotron");

}  // namespace katrin
