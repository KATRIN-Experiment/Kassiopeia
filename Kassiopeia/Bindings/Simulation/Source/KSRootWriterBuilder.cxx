#include "KSRootWriterBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSRootWriterBuilder::~KComplexElement() = default;

STATICINT sKSRootWriter = KSRootBuilder::ComplexElement<KSRootWriter>("ks_root_writer");

STATICINT sKSRootWriterStructure =
    KSRootWriterBuilder::Attribute<std::string>("name") + KSRootWriterBuilder::Attribute<std::string>("add_writer");

}  // namespace katrin
