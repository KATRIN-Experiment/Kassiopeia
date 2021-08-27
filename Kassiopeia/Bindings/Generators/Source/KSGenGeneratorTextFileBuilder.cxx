#include "KSGenGeneratorTextFileBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenGeneratorTextFileBuilder::~KComplexElement() = default;

STATICINT sKSGenGeneratorTextFileStructure =
    KSGenGeneratorTextFileBuilder::Attribute<std::string>("name") +
    KSGenGeneratorTextFileBuilder::Attribute<std::string>("base") +
    KSGenGeneratorTextFileBuilder::Attribute<std::string>("path");

STATICINT sKSGenGeneratorTextFile =
    KSRootBuilder::ComplexElement<KSGenGeneratorTextFile>("ksgen_generator_file");

}  // namespace katrin
