#include "KSComponentMaximumAtBuilder.h"

#include "KSComponentGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSComponentMaximumAtBuilder::~KComplexElement() = default;

STATICINT sKSComponentMaximumAtStructure =
    //! the name of this component.
    KSComponentMaximumAtBuilder::Attribute<std::string>("name") +
    KSComponentMaximumAtBuilder::Attribute<std::string>("group") +
    KSComponentMaximumAtBuilder::Attribute<std::string>("component") +
    //! the name of the output component.
    //! must be defined earlier in the configuration.
    KSComponentMaximumAtBuilder::Attribute<std::string>("parent") +
    //! the name of the analyzed source component.
    //! must be defined earlier in the configuration.
    KSComponentMaximumAtBuilder::Attribute<std::string>("source");

STATICINT sKSComponentMaximumAt =
    KSComponentGroupBuilder::ComplexElement<KSComponentMaximumAtData>("component_maximum_at") +
    KSComponentGroupBuilder::ComplexElement<KSComponentMaximumAtData>("output_maximum_at") +
    KSRootBuilder::ComplexElement<KSComponentMaximumAtData>("ks_component_maximum_at") +
    KSRootBuilder::ComplexElement<KSComponentMaximumAtData>("output_maximum_at");

}  // namespace katrin
