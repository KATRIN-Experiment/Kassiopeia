#include "KSWriteVTKBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    static int sKSWriteVTKStructure =
        KSWriteVTKBuilder::Attribute< string >( "name" ) +
        KSWriteVTKBuilder::Attribute< string >( "base" ) +
        KSWriteVTKBuilder::Attribute< string >( "path" );

    static int sKSWriteVTK =
        KSRootBuilder::ComplexElement< KSWriteVTK >( "kswrite_vtk" );
}
