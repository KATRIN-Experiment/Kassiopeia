/**
 * @file KGPlyFileBuilder.cc
 * @author Jan Behrens <jan.behrens@kit.edu>
 * @date 2022-11-24
 */

#include "KGPlyFileBuilder.hh"
#include "KGInterfaceBuilder.hh"

using namespace KGeoBag;
using namespace std;

namespace katrin
{

STATICINT sKGPlyFileBuilderStructure =
        KGPlyFileBuilder::Attribute<std::string>("file") +
        KGPlyFileBuilder::Attribute<std::string>("path") +
        KGPlyFileBuilder::Attribute<int>("mesh_count") +
        KGPlyFileBuilder::Attribute<double>("scale") +
        KGPlyFileBuilder::Attribute<string>("selector");

STATICINT sKGPlyFileSurfaceBuilderStructure =
        KGPlyFileSurfaceBuilder::Attribute<std::string>("name") +
        KGPlyFileSurfaceBuilder::ComplexElement<KGPlyFile>("ply_file");

STATICINT sKGPlyFileSurfaceBuilder = KGInterfaceBuilder::ComplexElement<KGWrappedSurface<KGPlyFile>>("ply_file_surface");

STATICINT sKGPlyFileSpaceBuilderStructure =
        KGPlyFileSpaceBuilder::Attribute<std::string>("name") +
        KGPlyFileSpaceBuilder::ComplexElement<KGPlyFile>("ply_file");

STATICINT sKGPlyFileSpaceBuilder = KGInterfaceBuilder::ComplexElement<KGWrappedSpace<KGPlyFile>>("ply_file_space");

}  // namespace katrin
