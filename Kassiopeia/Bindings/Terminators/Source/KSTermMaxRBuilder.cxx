#include "KSTermMaxRBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMaxRBuilder::~KComplexElement() = default;

STATICINT sKSTermMaxRStructure =
    //! the name of the terminator.
    KSTermMaxRBuilder::Attribute<std::string>("name") +
    //! the maximum allowed radial coordinate.
    KSTermMaxRBuilder::Attribute<double>("r");

STATICINT sKSTermMaxR = KSRootBuilder::ComplexElement<KSTermMaxR>("ksterm_max_r");

}  // namespace katrin
