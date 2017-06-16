#include "KSWriteVTKBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    STATICINT sKSWriteVTKStructure =
        KSWriteVTKBuilder::Attribute< string >( "name" ) +
        KSWriteVTKBuilder::Attribute< string >( "base" ) +
        KSWriteVTKBuilder::Attribute< string >( "path" );

    STATICINT sKSWriteVTK =
        KSRootBuilder::ComplexElement< KSWriteVTK >( "kswrite_vtk" );
}
