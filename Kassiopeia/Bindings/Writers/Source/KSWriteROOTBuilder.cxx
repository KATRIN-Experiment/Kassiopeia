#include "KSWriteROOTBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSWriteROOTBuilder::~KComplexElement() = default;

STATICINT sKSWriteROOTStructure =
    //! the name of this writer.
    KSWriteROOTBuilder::Attribute<std::string>("name") +
    //! base name of the file.
    KSWriteROOTBuilder::Attribute<std::string>("base") +
    //! path where the file should be written.
    KSWriteROOTBuilder::Attribute<std::string>("path");

STATICINT sKSWriteROOT = KSRootBuilder::ComplexElement<KSWriteROOT>("kswrite_root");

}  // namespace katrin
