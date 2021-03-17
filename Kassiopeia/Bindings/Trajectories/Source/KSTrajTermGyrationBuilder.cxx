#include "KSTrajTermGyrationBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajTermGyrationBuilder::~KComplexElement() = default;

STATICINT sKSTrajTermGyrationStructure = KSTrajTermGyrationBuilder::Attribute<std::string>("name");

STATICINT sToolboxKSTrajTermGyration = KSRootBuilder::ComplexElement<KSTrajTermGyration>("kstraj_term_gyration");

}  // namespace katrin
