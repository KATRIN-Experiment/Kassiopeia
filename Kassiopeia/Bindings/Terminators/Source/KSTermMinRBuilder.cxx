#include "KSTermMinRBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMinRBuilder::~KComplexElement() = default;

STATICINT sKSTermMinRStructure =
    KSTermMinRBuilder::Attribute<std::string>("name") + KSTermMinRBuilder::Attribute<double>("r");

STATICINT sKSTermMinR = KSRootBuilder::ComplexElement<KSTermMinR>("ksterm_min_r");

}  // namespace katrin
