#include "KSWriteVTKBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

STATICINT sKSWriteVTKStructure =
    //! the name of this writer.
    KSWriteVTKBuilder::Attribute<std::string>("name") +
    //! base name of the file.
    KSWriteVTKBuilder::Attribute<std::string>("base") +
    //! path where the file should be written.
    KSWriteVTKBuilder::Attribute<std::string>("path");

STATICINT sKSWriteVTK = KSRootBuilder::ComplexElement<KSWriteVTK>("kswrite_vtk");
}  // namespace katrin
