#include "KSTermDeathBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermDeathBuilder::~KComplexElement() = default;

STATICINT sKSTermDeathStructure = KSTermDeathBuilder::Attribute<std::string>("name");

STATICINT sKSTermDeath = KSRootBuilder::ComplexElement<KSTermDeath>("ksterm_death");


}  // namespace katrin
