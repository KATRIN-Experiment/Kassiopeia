#include "KElectrostaticExpoFieldBuilder.hh"

#include "KEMToolboxBuilder.hh"


using namespace KEMField;
namespace katrin {

template<> KElectrostaticExpoFieldBuilder::~KComplexElement() = default;

STATICINT sKElectrostaticExpoField =
    KEMToolboxBuilder::ComplexElement< KElectrostaticExpoField >( "expo_electric_field" );

STATICINT sKElectrostaticExpoFieldStructure =
    KElectrostaticExpoFieldBuilder::Attribute<std::string>( "name" ) +
    KElectrostaticExpoFieldBuilder::Attribute<double>("TKE") +
    KElectrostaticExpoFieldBuilder::Attribute<double>("B0") +
    KElectrostaticExpoFieldBuilder::Attribute<double>("lambda") +
    KElectrostaticExpoFieldBuilder::Attribute<double>("Y0");

//    KElectrostaticExpoFieldBuilder::Attribute<KEMStreamableThreeVector>( "Ey" ) +
//    KElectrostaticExpoFieldBuilder::Attribute<KEMStreamableThreeVector>( "Epar" );

} //katrin



