#include "KSTrajTermGravityBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajTermGravityBuilder::~KComplexElement() = default;

STATICINT sKSTrajTermGravityStructure = KSTrajTermGravityBuilder::Attribute<std::string>("name") +
                                        KSTrajTermGravityBuilder::Attribute<KThreeVector>("gravity");

STATICINT sToolboxKSTrajTermGravity = KSRootBuilder::ComplexElement<KSTrajTermGravity>("kstraj_term_gravity");

}  // namespace katrin
