#include "KSTrajIntegratorRK87Builder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajIntegratorRK87Builder::~KComplexElement() = default;

STATICINT sKSTrajIntegratorRK87Structure = KSTrajIntegratorRK87Builder::Attribute<std::string>("name");

STATICINT sToolboxKSTrajIntegratorRK87 = KSRootBuilder::ComplexElement<KSTrajIntegratorRK87>("kstraj_integrator_rk87");


}  // namespace katrin
