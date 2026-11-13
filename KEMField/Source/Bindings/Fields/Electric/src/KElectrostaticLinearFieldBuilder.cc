#include "KElectrostaticLinearFieldBuilder.hh"

#include "KEMToolboxBuilder.hh"


using namespace KEMField;
namespace katrin
{

template<> KElectrostaticLinearFieldBuilder::~KComplexElement() = default;

STATICINT sKEMToolBoxBuilder = KEMToolboxBuilder::ComplexElement<KElectrostaticLinearField>("linear_electric_field");

STATICINT sKElectrostaticConstantFieldBuilder =
    //! the name of the electric field.
    KElectrostaticLinearFieldBuilder::Attribute<std::string>("name") +
    //! z0 in the equation of the element description, in meters.
    KElectrostaticLinearFieldBuilder::Attribute<double>("U1") +
    //! z1 in the equation of the element description, in meters.
    KElectrostaticLinearFieldBuilder::Attribute<double>("U2") +
    //! V0 in the equation of the element description, in volts.
    KElectrostaticLinearFieldBuilder::Attribute<double>("z1") +
    //! V1 in the equation of the element description, in volts.
    KElectrostaticLinearFieldBuilder::Attribute<double>("z2") +
    KElectrostaticLinearFieldBuilder::Attribute<std::string>("surface");  // TODO

}  // namespace katrin
