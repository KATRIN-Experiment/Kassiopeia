#include "KSWriteVTKBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

STATICINT sKSWriteVTKStructure = KSWriteVTKBuilder::Attribute<std::string>("name") +
                                 KSWriteVTKBuilder::Attribute<std::string>("base") +
                                 KSWriteVTKBuilder::Attribute<std::string>("path");

STATICINT sKSWriteVTK = KSRootBuilder::ComplexElement<KSWriteVTK>("kswrite_vtk");
}  // namespace katrin
