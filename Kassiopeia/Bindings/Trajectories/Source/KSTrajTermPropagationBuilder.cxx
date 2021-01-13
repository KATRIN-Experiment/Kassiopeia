#include "KSTrajTermPropagationBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajTermPropagationBuilder::~KComplexElement() = default;

STATICINT sKSTrajTermPropagationStructure = KSTrajTermPropagationBuilder::Attribute<std::string>("name") +
                                            KSTrajTermPropagationBuilder::Attribute<std::string>("direction");

STATICINT sToolboxKSTrajTermPropagation =
    KSRootBuilder::ComplexElement<KSTrajTermPropagation>("kstraj_term_propagation");

}  // namespace katrin
