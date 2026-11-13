#include "KSTermMaxTimeBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMaxTimeBuilder::~KComplexElement() = default;

STATICINT sKSTermMaxTimeStructure =
    //! the name of the terminator.
    KSTermMaxTimeBuilder::Attribute<std::string>("name") +
    //! the maximum allowed track time.
    KSTermMaxTimeBuilder::Attribute<double>("time");

STATICINT sKSTermMaxTime = KSRootBuilder::ComplexElement<KSTermMaxTime>("ksterm_max_time");

}  // namespace katrin
