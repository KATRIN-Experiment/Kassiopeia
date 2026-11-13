#include "KSTermMaxZBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMaxZBuilder::~KComplexElement() = default;

STATICINT sKSTermMaxZStructure =
    //! the name of the terminator.
    KSTermMaxZBuilder::Attribute<std::string>("name") +
    //! the maximum allowed z coordinate.
    KSTermMaxZBuilder::Attribute<double>("z");

STATICINT sToolboxKSTermMaxZ = KSRootBuilder::ComplexElement<KSTermMaxZ>("ksterm_max_z");


}  // namespace katrin
