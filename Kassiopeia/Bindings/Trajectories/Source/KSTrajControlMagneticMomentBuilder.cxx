#include "KSTrajControlMagneticMomentBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajControlMagneticMomentBuilder::~KComplexElement() = default;

STATICINT sKSTrajControlMagneticMomentStructure = KSTrajControlMagneticMomentBuilder::Attribute<std::string>("name") +
                                                  KSTrajControlMagneticMomentBuilder::Attribute<double>("lower_limit") +
                                                  KSTrajControlMagneticMomentBuilder::Attribute<double>("upper_limit");

STATICINT sToolboxKSTrajControlMagneticMoment =
    KSRootBuilder::ComplexElement<KSTrajControlMagneticMoment>("kstraj_control_magnetic_moment");

}  // namespace katrin
