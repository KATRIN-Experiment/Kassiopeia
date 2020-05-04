#include "KESSInelasticBetheFanoBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KESSInelasticBetheFanoBuilder::~KComplexElement() {}

STATICINT sKSInelasticBetheFanoStructure = KESSInelasticBetheFanoBuilder::Attribute<string>("name") +
                                           KESSInelasticBetheFanoBuilder::Attribute<bool>("PhotoAbsorption") +
                                           KESSInelasticBetheFanoBuilder::Attribute<bool>("AugerRelaxation");

STATICINT sKSInelasticBetheFano = KSRootBuilder::ComplexElement<KESSInelasticBetheFano>("kess_inelastic_bethefano");

}  // namespace katrin
