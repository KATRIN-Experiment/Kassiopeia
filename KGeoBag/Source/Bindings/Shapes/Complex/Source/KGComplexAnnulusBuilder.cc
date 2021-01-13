#include "KGComplexAnnulusBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGComplexAnnulusRingBuilderStructure = KGComplexAnnulusRingBuilder::Attribute<double>("radius") +
                                                  KGComplexAnnulusRingBuilder::Attribute<double>("x") +
                                                  KGComplexAnnulusRingBuilder::Attribute<double>("y");

STATICINT sKGComplexAnnulusBuilderStructure = KGComplexAnnulusBuilder::Attribute<double>("radius") +
                                              KGComplexAnnulusBuilder::Attribute<int>("radial_mesh_count") +
                                              KGComplexAnnulusBuilder::Attribute<int>("axial_mesh_count") +
                                              KGComplexAnnulusBuilder::ComplexElement<KGComplexAnnulus::Ring>("ring");

STATICINT sKGComplexAnnulusSurfaceBuilderStructure =
    KGComplexAnnulusSurfaceBuilder::Attribute<std::string>("name") +
    KGComplexAnnulusSurfaceBuilder::ComplexElement<KGComplexAnnulus>("complex_annulus");

STATICINT sKGComplexAnnulusSurfaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSurface<KGComplexAnnulus>>("complex_annulus_surface");

}  // namespace katrin
