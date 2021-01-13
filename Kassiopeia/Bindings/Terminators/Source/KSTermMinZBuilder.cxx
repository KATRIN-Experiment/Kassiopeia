#include "KSTermMinZBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMinZBuilder::~KComplexElement() = default;

STATICINT sKSTermMinZStructure =
    KSTermMinZBuilder::Attribute<std::string>("name") + KSTermMinZBuilder::Attribute<double>("z");

STATICINT sKSTermMinZ = KSRootBuilder::ComplexElement<KSTermMinZ>("ksterm_min_z");


}  // namespace katrin
