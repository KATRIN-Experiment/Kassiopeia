#include "KGInterfaceBuilder.hh"

#include "KElementProcessor.hh"
#include "KRoot.h"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

template<> KGInterfaceBuilder::~KComplexElement() {}

STATICINT sKGInterfaceStructure = KGInterfaceBuilder::Attribute<bool>("reset");

STATICINT sKGInterface = KRootBuilder::ComplexElement<KGInterface>("geometry");

STATICINT sKGInterfaceCompat = KElementProcessor::ComplexElement<KGInterface>("geometry");
}  // namespace katrin
