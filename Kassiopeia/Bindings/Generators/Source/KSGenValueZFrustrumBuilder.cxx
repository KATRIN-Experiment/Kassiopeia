#include "KSGenValueZFrustrumBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueZFrustrumBuilder::~KComplexElement() = default;

STATICINT sKSGenValueZFrustrumStructure =
    KSGenValueZFrustrumBuilder::Attribute<std::string>("name") + KSGenValueZFrustrumBuilder::Attribute<double>("r1") +
    KSGenValueZFrustrumBuilder::Attribute<double>("r2") + KSGenValueZFrustrumBuilder::Attribute<double>("z1") +
    KSGenValueZFrustrumBuilder::Attribute<double>("z2");

STATICINT sToolboxKSGenValueZFrustrum = KSRootBuilder::ComplexElement<KSGenValueZFrustrum>("ksgen_value_z_frustrum");

}  // namespace katrin
