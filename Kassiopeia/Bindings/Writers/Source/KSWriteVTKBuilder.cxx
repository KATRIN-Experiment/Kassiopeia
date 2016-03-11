#include "KSWriteVTKBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    STATICINT sKSWriteVTKStructure =
        KSWriteVTKBuilder::Attribute< string >( "name" ) +
        KSWriteVTKBuilder::Attribute< string >( "base" ) +
        KSWriteVTKBuilder::Attribute< string >( "path" );

    STATICINT sKSWriteVTK =
        KSRootBuilder::ComplexElement< KSWriteVTK >( "kswrite_vtk" );
}
