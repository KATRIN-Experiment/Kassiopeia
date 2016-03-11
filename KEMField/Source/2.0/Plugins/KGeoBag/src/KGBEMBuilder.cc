#include "KGBEMBuilder.hh"
#include "KGInterfaceBuilder.hh"

using namespace KGeoBag;
namespace katrin
{

    STATICINT sElectrostaticDirichlet =
        KGInterfaceBuilder::ComplexElement< KGBEMAttributor< KEMField::KElectrostaticBasis, KEMField::KDirichletBoundary > >( "electrostatic_dirichlet" );

    STATICINT sElectrostaticDirichletStructure =
        KGElectrostaticDirichletBuilder::Attribute< string >( "name" ) +
        KGElectrostaticDirichletBuilder::Attribute< double >( "value" ) +
        KGElectrostaticDirichletBuilder::Attribute< string >( "surfaces" ) +
        KGElectrostaticDirichletBuilder::Attribute< string >( "spaces" );

    STATICINT sElectrostaticNeumann =
        KGInterfaceBuilder::ComplexElement< KGBEMAttributor< KEMField::KElectrostaticBasis, KEMField::KNeumannBoundary > >( "electrostatic_neumann" );

    STATICINT sElectrostaticNeumannStructure =
        KGElectrostaticNeumannBuilder::Attribute< string >( "name" ) +
        KGElectrostaticNeumannBuilder::Attribute< double >( "flux" ) +
        KGElectrostaticNeumannBuilder::Attribute< string >( "surfaces" ) +
        KGElectrostaticNeumannBuilder::Attribute< string >( "spaces" );

    //STATICINT sMagnetostaticDirichlet =
    //    KGInterfaceBuilder::ComplexElement< KGBEMAttributor< KEMField::KMagnetostaticBasis, KEMField::KDirichletBoundary > >( "magnetostatic_dirichlet" );

    //STATICINT sMagnetostaticNeumann =
    //    KGInterfaceBuilder::ComplexElement< KGBEMAttributor< KEMField::KMagnetostaticBasis, KEMField::KNeumannBoundary > >( "magnetostatic_neumann" );

}
