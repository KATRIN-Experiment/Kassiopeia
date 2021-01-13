#include "KGTransformationBuilder.hh"

namespace katrin
{

template<> KGTransformationBuilder::~KComplexElement() = default;

STATICINT sTransformationBuilderStructure = KGTransformationBuilder::Attribute<KThreeVector>("displacement") +
                                            KGTransformationBuilder::Attribute<KThreeVector>("d") +
                                            KGTransformationBuilder::Attribute<KThreeVector>("rotation_euler") +
                                            KGTransformationBuilder::Attribute<KThreeVector>("r_eu") +
                                            KGTransformationBuilder::Attribute<KThreeVector>("rotation_axis_angle") +
                                            KGTransformationBuilder::Attribute<KThreeVector>("r_aa");

}  // namespace katrin
