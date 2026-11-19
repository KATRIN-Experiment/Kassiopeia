#include "KSTermSecondariesBuilder.h"

#include "KSRootBuilder.h"


using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermSecondariesBuilder::~KComplexElement() = default;

//! the name of the terminator.
STATICINT sKSTermSecondariesStructure = KSTermSecondariesBuilder::Attribute<std::string>("name");

STATICINT sToolboxKSTermSecondaries = KSRootBuilder::ComplexElement<KSTermSecondaries>("ksterm_secondaries");

}  // namespace katrin
