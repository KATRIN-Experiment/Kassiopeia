#include "KGSpaceBuilder.hh"

#include "KGInterfaceBuilder.hh"
#include "KGSurfaceBuilder.hh"
#include "KGTransformationBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{
STATICINT sKGSpaceBuilder = KGInterfaceBuilder::ComplexElement<KGSpace>("space");

STATICINT sKGSpaceBuilderStructure =
    KGSpaceBuilder::Attribute<std::string>("name") + KGSpaceBuilder::Attribute<std::string>("node") +
    KGSpaceBuilder::Attribute<std::string>("tree") + KGSpaceBuilder::ComplexElement<KTransformation>("transformation") +
    KGSpaceBuilder::ComplexElement<KGSpace>("space") + KGSpaceBuilder::ComplexElement<KGSurface>("surface");
}  // namespace katrin
