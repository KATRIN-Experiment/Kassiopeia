#include "KSTermMinDistanceBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMinDistanceBuilder::~KComplexElement() {}

STATICINT sKSTermMinDistanceStructure = KSTermMinDistanceBuilder::Attribute<string>("name") +
                                        KSTermMinDistanceBuilder::Attribute<string>("surfaces") +
                                        KSTermMinDistanceBuilder::Attribute<double>("min_distance");

STATICINT sKSTermMinDistance = KSRootBuilder::ComplexElement<KSTermMinDistance>("ksterm_min_distance");

}  // namespace katrin
