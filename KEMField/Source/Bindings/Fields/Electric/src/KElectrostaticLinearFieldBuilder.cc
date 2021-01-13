#include "KElectrostaticLinearFieldBuilder.hh"

#include "KEMToolboxBuilder.hh"


using namespace KEMField;
namespace katrin
{

template<> KElectrostaticLinearFieldBuilder::~KComplexElement() = default;

STATICINT sKEMToolBoxBuilder = KEMToolboxBuilder::ComplexElement<KElectrostaticLinearField>("linear_electric_field");

STATICINT sKElectrostaticConstantFieldBuilder =
    KElectrostaticLinearFieldBuilder::Attribute<std::string>("name") +
    KElectrostaticLinearFieldBuilder::Attribute<double>("U1") +
    KElectrostaticLinearFieldBuilder::Attribute<double>("U2") +
    KElectrostaticLinearFieldBuilder::Attribute<double>("z1") +
    KElectrostaticLinearFieldBuilder::Attribute<double>("z2") +
    KElectrostaticLinearFieldBuilder::Attribute<std::string>("surface");  // TODO

}  // namespace katrin
