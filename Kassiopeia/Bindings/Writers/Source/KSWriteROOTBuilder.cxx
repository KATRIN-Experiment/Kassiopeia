#include "KSWriteROOTBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSWriteROOTBuilder::~KComplexElement() = default;

STATICINT sKSWriteROOTStructure = KSWriteROOTBuilder::Attribute<std::string>("name") +
                                  KSWriteROOTBuilder::Attribute<std::string>("base") +
                                  KSWriteROOTBuilder::Attribute<std::string>("path");

STATICINT sKSWriteROOT = KSRootBuilder::ComplexElement<KSWriteROOT>("kswrite_root");

}  // namespace katrin
