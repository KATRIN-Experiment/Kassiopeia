#include "KSWriteASCIIBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSWriteASCIIBuilder::~KComplexElement() = default;

STATICINT sKSWriteASCIIStructure =
    KSWriteASCIIBuilder::Attribute<std::string>("name") + KSWriteASCIIBuilder::Attribute<std::string>("base") +
    KSWriteASCIIBuilder::Attribute<std::string>("path") + KSWriteASCIIBuilder::Attribute<unsigned int>("precision");

STATICINT sKSWriteASCII = KSRootBuilder::ComplexElement<KSWriteASCII>("kswrite_ascii");

}  // namespace katrin
