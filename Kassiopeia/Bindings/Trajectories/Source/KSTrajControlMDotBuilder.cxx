#include "KSTrajControlMDotBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajControlMDotBuilder::~KComplexElement() = default;

STATICINT sKSTrajControlMDotStructure =
    KSTrajControlMDotBuilder::Attribute<std::string>("name") + KSTrajControlMDotBuilder::Attribute<double>("fraction");

STATICINT sToolboxKSTrajControlMDot = KSRootBuilder::ComplexElement<KSTrajControlMDot>("kstraj_control_m_dot");

}  // namespace katrin
