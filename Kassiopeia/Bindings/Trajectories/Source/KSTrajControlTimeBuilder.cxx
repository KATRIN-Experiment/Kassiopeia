#include "KSTrajControlTimeBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajControlTimeBuilder::~KComplexElement() = default;

STATICINT sKSTrajControlTimeStructure =
    KSTrajControlTimeBuilder::Attribute<std::string>("name") + KSTrajControlTimeBuilder::Attribute<double>("time");

STATICINT sToolboxKSTrajControlTime = KSRootBuilder::ComplexElement<KSTrajControlTime>("kstraj_control_time");

}  // namespace katrin
