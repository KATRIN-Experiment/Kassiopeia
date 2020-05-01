#include "KESSElasticElsepaBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KESSElasticElsepaBuilder::~KComplexElement() {}

STATICINT sKSElasticElsepaStructure = KESSElasticElsepaBuilder::Attribute<string>("name");

STATICINT sKSElasticElsepa = KSRootBuilder::ComplexElement<KESSElasticElsepa>("kess_elastic_elsepa");

}  // namespace katrin
