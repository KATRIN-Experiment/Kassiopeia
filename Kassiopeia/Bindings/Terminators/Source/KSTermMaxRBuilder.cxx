#include "KSTermMaxRBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMaxRBuilder::~KComplexElement() = default;

STATICINT sKSTermMaxRStructure =
    KSTermMaxRBuilder::Attribute<std::string>("name") + KSTermMaxRBuilder::Attribute<double>("r");

STATICINT sKSTermMaxR = KSRootBuilder::ComplexElement<KSTermMaxR>("ksterm_max_r");

}  // namespace katrin
