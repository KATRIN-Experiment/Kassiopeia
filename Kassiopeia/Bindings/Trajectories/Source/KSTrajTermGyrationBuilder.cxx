#include "KSTrajTermGyrationBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajTermGyrationBuilder::~KComplexElement() {}

STATICINT sKSTrajTermGyrationStructure = KSTrajTermGyrationBuilder::Attribute<string>("name");

STATICINT sToolboxKSTrajTermGyration = KSRootBuilder::ComplexElement<KSTrajTermGyration>("kstraj_term_gyration");

}  // namespace katrin
