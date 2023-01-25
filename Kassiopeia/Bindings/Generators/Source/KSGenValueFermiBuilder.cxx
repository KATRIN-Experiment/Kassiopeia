#include "KSGenValueFermiBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueFermiBuilder::~KComplexElement() = default;

STATICINT sKSGenValueFermiStructure = KSGenValueFermiBuilder::Attribute<std::string>("name") +
                                                 KSGenValueFermiBuilder::Attribute<double>("value_min") +
                                                 KSGenValueFermiBuilder::Attribute<double>("value_max") +
                                                 KSGenValueFermiBuilder::Attribute<double>("value_mean") +
                                                 KSGenValueFermiBuilder::Attribute<double>("value_tau") +
                                                 KSGenValueFermiBuilder::Attribute<double>("value_temp");

STATICINT sKSGenValueFermi =
    KSRootBuilder::ComplexElement<KSGenValueFermi>("ksgen_value_fermi");

}  // namespace katrin
