#include "KGBEMBuilder.hh"
#include "KGInterfaceBuilder.hh"

using namespace KGeoBag;
namespace katrin
{

    static const int sElectrostaticDirichlet =
        KGInterfaceBuilder::ComplexElement< KGBEMAttributor< KEMField::KElectrostaticBasis, KEMField::KDirichletBoundary > >( "electrostatic_dirichlet" );

    static const int sElectrostaticDirichletStructure =
        KGElectrostaticDirichletBuilder::Attribute< string >( "name" ) +
        KGElectrostaticDirichletBuilder::Attribute< double >( "value" ) +
        KGElectrostaticDirichletBuilder::Attribute< string >( "surfaces" ) +
        KGElectrostaticDirichletBuilder::Attribute< string >( "spaces" );

    static const int sElectrostaticNeumann =
        KGInterfaceBuilder::ComplexElement< KGBEMAttributor< KEMField::KElectrostaticBasis, KEMField::KNeumannBoundary > >( "electrostatic_neumann" );

    static const int sElectrostaticNeumannStructure =
        KGElectrostaticNeumannBuilder::Attribute< string >( "name" ) +
        KGElectrostaticNeumannBuilder::Attribute< double >( "flux" ) +
        KGElectrostaticNeumannBuilder::Attribute< string >( "surfaces" ) +
        KGElectrostaticNeumannBuilder::Attribute< string >( "spaces" );

    //static const int sMagnetostaticDirichlet =
    //    KGInterfaceBuilder::ComplexElement< KGBEMAttributor< KEMField::KMagnetostaticBasis, KEMField::KDirichletBoundary > >( "magnetostatic_dirichlet" );

    //static const int sMagnetostaticNeumann =
    //    KGInterfaceBuilder::ComplexElement< KGBEMAttributor< KEMField::KMagnetostaticBasis, KEMField::KNeumannBoundary > >( "magnetostatic_neumann" );

}
