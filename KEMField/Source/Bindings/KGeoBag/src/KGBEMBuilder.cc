#include "KGBEMBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace KGeoBag;
using namespace std;

namespace katrin
{

STATICINT sElectrostaticDirichlet =
    KGInterfaceBuilder::ComplexElement<KGBEMAttributor<KEMField::KElectrostaticBasis, KEMField::KDirichletBoundary>>(
        "electrostatic_dirichlet");

STATICINT sElectrostaticDirichletStructure = KGElectrostaticDirichletBuilder::Attribute<std::string>("name") +
                                             KGElectrostaticDirichletBuilder::Attribute<double>("value") +
                                             KGElectrostaticDirichletBuilder::Attribute<std::string>("surfaces") +
                                             KGElectrostaticDirichletBuilder::Attribute<std::string>("spaces");

STATICINT sElectrostaticNeumann =
    KGInterfaceBuilder::ComplexElement<KGBEMAttributor<KEMField::KElectrostaticBasis, KEMField::KNeumannBoundary>>(
        "electrostatic_neumann");

STATICINT sElectrostaticNeumannStructure = KGElectrostaticNeumannBuilder::Attribute<std::string>("name") +
                                           KGElectrostaticNeumannBuilder::Attribute<double>("flux") +
                                           KGElectrostaticNeumannBuilder::Attribute<std::string>("surfaces") +
                                           KGElectrostaticNeumannBuilder::Attribute<std::string>("spaces");

//STATICINT sMagnetostaticDirichlet =
//    KGInterfaceBuilder::ComplexElement< KGBEMAttributor< KEMField::KMagnetostaticBasis, KEMField::KDirichletBoundary > >( "magnetostatic_dirichlet" );

//STATICINT sMagnetostaticNeumann =
//    KGInterfaceBuilder::ComplexElement< KGBEMAttributor< KEMField::KMagnetostaticBasis, KEMField::KNeumannBoundary > >( "magnetostatic_neumann" );

}  // namespace katrin
