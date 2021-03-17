#include "KGSurfaceBuilder.hh"

#include "KGInterfaceBuilder.hh"
#include "KGTransformationBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{
STATICINT sKGSurfaceBuilder = KGInterfaceBuilder::ComplexElement<KGSurface>("surface");

STATICINT sKGSurfaceBuilderStructure = KGSurfaceBuilder::Attribute<std::string>("name") +
                                       KGSurfaceBuilder::Attribute<std::string>("node") +
                                       KGSurfaceBuilder::ComplexElement<KTransformation>("transformation");
}  // namespace katrin
