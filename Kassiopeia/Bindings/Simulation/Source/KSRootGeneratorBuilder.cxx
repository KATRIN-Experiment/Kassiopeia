#include "KSRootGeneratorBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSRootGeneratorBuilder::~KComplexElement() {}

STATICINT sKSRootGenerator = KSRootBuilder::ComplexElement<KSRootGenerator>("ks_root_generator");

STATICINT sKSRootGeneratorStructure =
    KSRootGeneratorBuilder::Attribute<string>("name") + KSRootGeneratorBuilder::Attribute<string>("set_generator");

}  // namespace katrin
