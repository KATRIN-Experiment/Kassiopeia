#include "KSTrajControlLengthBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajControlLengthBuilder::~KComplexElement() = default;

STATICINT sKSTrajControlLengthStructure = KSTrajControlLengthBuilder::Attribute<std::string>("name") +
                                          KSTrajControlLengthBuilder::Attribute<double>("length");

STATICINT sToolboxKSTrajControlLength = KSRootBuilder::ComplexElement<KSTrajControlLength>("kstraj_control_length");

}  // namespace katrin
