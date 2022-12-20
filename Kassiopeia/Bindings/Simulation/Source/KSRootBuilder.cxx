#include "KSRootBuilder.h"

#include "KElementProcessor.hh"
#include "KRoot.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSRootBuilder::~KComplexElement() = default;

STATICINT sKSRootStructure = KSRootBuilder::Attribute<unsigned int>("random_seed");

STATICINT sKSRoot = KRootBuilder::ComplexElement<KSRoot>("kassiopeia");

STATICINT sKSRootCompat = KElementProcessor::ComplexElement<KSRoot>("kassiopeia");
}  // namespace katrin
