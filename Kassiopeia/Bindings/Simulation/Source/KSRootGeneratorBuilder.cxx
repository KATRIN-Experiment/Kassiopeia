#include "KSRootGeneratorBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSRootGeneratorBuilder::~KComplexElement() = default;

STATICINT sKSRootGenerator = KSRootBuilder::ComplexElement<KSRootGenerator>("ks_root_generator");

STATICINT sKSRootGeneratorStructure = KSRootGeneratorBuilder::Attribute<std::string>("name") +
                                      KSRootGeneratorBuilder::Attribute<std::string>("set_generator");

}  // namespace katrin
