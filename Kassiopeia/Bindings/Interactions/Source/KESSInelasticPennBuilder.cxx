#include "KESSInelasticPennBuilder.h"

#include "KSRootBuilder.h"


using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KESSInelasticPennBuilder::~KComplexElement() = default;

STATICINT sKSInelasticPennStructure = KESSInelasticPennBuilder::Attribute<std::string>("name") +
                                      KESSInelasticPennBuilder::Attribute<bool>("PhotoAbsorption") +
                                      KESSInelasticPennBuilder::Attribute<bool>("AugerRelaxation");

STATICINT sKSInelasticPenn = KSRootBuilder::ComplexElement<KESSInelasticPenn>("kess_inelastic_penn");

}  // namespace katrin
