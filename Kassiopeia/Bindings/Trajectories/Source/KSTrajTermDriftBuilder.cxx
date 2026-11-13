#include "KSTrajTermDriftBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajTermDriftBuilder::~KComplexElement() = default;

//! the name of this term.
STATICINT sKSTrajTermDriftStructure = KSTrajTermDriftBuilder::Attribute<std::string>("name");

STATICINT sToolboxKSTrajTermDrift = KSRootBuilder::ComplexElement<KSTrajTermDrift>("kstraj_term_drift");

}  // namespace katrin
