#include "KSTrajIntegratorRKDP853Builder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajIntegratorRKDP853Builder::~KComplexElement() = default;

STATICINT sKSTrajIntegratorRKDP853Structure = KSTrajIntegratorRKDP853Builder::Attribute<std::string>("name");

STATICINT sToolboxKSTrajIntegratorRKDP853 =
    KSRootBuilder::ComplexElement<KSTrajIntegratorRKDP853>("kstraj_integrator_rkdp853");

}  // namespace katrin
