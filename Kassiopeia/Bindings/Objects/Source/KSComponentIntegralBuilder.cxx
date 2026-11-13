#include "KSComponentIntegralBuilder.h"

#include "KSComponentGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSComponentIntegralBuilder::~KComplexElement() = default;

STATICINT sKSComponentIntegralStructure =
    //! the name of this component.
    KSComponentIntegralBuilder::Attribute<std::string>("name") +
    KSComponentIntegralBuilder::Attribute<std::string>("group") +
    KSComponentIntegralBuilder::Attribute<std::string>("component") +
    //! the name of the integrand.
    //! must be defined earlier in the configuration.
    KSComponentIntegralBuilder::Attribute<std::string>("parent");

STATICINT sKSComponentIntegral =
    KSComponentGroupBuilder::ComplexElement<KSComponentIntegralData>("component_integral") +
    KSComponentGroupBuilder::ComplexElement<KSComponentIntegralData>("output_integral") +
    KSRootBuilder::ComplexElement<KSComponentIntegralData>("ks_component_integral") +
    KSRootBuilder::ComplexElement<KSComponentIntegralData>("output_integral");


}  // namespace katrin
