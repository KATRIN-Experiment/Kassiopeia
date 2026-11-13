#include "KElectrostaticConstantFieldBuilder.hh"

#include "KEMToolboxBuilder.hh"


using namespace KEMField;
namespace katrin
{

template<> KElectrostaticConstantFieldBuilder::~KComplexElement() = default;

STATICINT sKEMToolBoxBuilder =
    KEMToolboxBuilder::ComplexElement<KElectrostaticConstantField>("constant_electric_field");

STATICINT sKElectrostaticConstantFieldBuilder =
    //! the name of the electric field.
    KElectrostaticConstantFieldBuilder::Attribute<std::string>("name") +
    //! the cartesian components of the electric field vector in volts per meter.
    KElectrostaticConstantFieldBuilder::Attribute<KEMStreamableThreeVector>("field") +
    KElectrostaticConstantFieldBuilder::Attribute<KEMStreamableThreeVector>("location") +
    KElectrostaticConstantFieldBuilder::Attribute<double>("offset_potential");

}  // namespace katrin
