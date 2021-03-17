#include "KESSElasticElsepaBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KESSElasticElsepaBuilder::~KComplexElement() = default;

STATICINT sKSElasticElsepaStructure = KESSElasticElsepaBuilder::Attribute<std::string>("name");

STATICINT sKSElasticElsepa = KSRootBuilder::ComplexElement<KESSElasticElsepa>("kess_elastic_elsepa");

}  // namespace katrin
