#include "KSWriteASCIIBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSWriteASCIIBuilder::~KComplexElement() {}

STATICINT sKSWriteASCIIStructure =
    KSWriteASCIIBuilder::Attribute<string>("name") + KSWriteASCIIBuilder::Attribute<string>("base") +
    KSWriteASCIIBuilder::Attribute<string>("path") + KSWriteASCIIBuilder::Attribute<unsigned int>("precision");

STATICINT sKSWriteASCII = KSRootBuilder::ComplexElement<KSWriteASCII>("kswrite_ascii");

}  // namespace katrin
