#include "KSTermMinDistanceBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMinDistanceBuilder::~KComplexElement() = default;

STATICINT sKSTermMinDistanceStructure = KSTermMinDistanceBuilder::Attribute<std::string>("name") +
                                        KSTermMinDistanceBuilder::Attribute<std::string>("surfaces") +
                                        KSTermMinDistanceBuilder::Attribute<double>("min_distance");

STATICINT sKSTermMinDistance = KSRootBuilder::ComplexElement<KSTermMinDistance>("ksterm_min_distance");

}  // namespace katrin
