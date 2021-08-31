/**
 * @file KGStlFileBuilder.cc
 * @author Jan Behrens <jan.behrens@kit.edu>
 * @date 2021-07-02
 */

#include "KGStlFileBuilder.hh"
#include "KGInterfaceBuilder.hh"

using namespace KGeoBag;
using namespace std;

namespace katrin
{

STATICINT sKGStlFileBuilderStructure =
        KGStlFileBuilder::Attribute<std::string>("file") +
        KGStlFileBuilder::Attribute<std::string>("path") +
        KGStlFileBuilder::Attribute<int>("mesh_count") +
        KGStlFileBuilder::Attribute<double>("scale") +
        KGStlFileBuilder::Attribute<string>("selector");

STATICINT sKGStlFileSurfaceBuilderStructure =
        KGStlFileSurfaceBuilder::Attribute<std::string>("name") +
        KGStlFileSurfaceBuilder::ComplexElement<KGStlFile>("stl_file");

STATICINT sKGStlFileSurfaceBuilder = KGInterfaceBuilder::ComplexElement<KGWrappedSurface<KGStlFile>>("stl_file_surface");

STATICINT sKGStlFileSpaceBuilderStructure =
        KGStlFileSpaceBuilder::Attribute<std::string>("name") +
        KGStlFileSpaceBuilder::ComplexElement<KGStlFile>("stl_file");

STATICINT sKGStlFileSpaceBuilder = KGInterfaceBuilder::ComplexElement<KGWrappedSpace<KGStlFile>>("stl_file_space");

}  // namespace katrin
