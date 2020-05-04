#include "KSTrajTermSynchrotronBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajTermSynchrotronBuilder::~KComplexElement() {}

STATICINT sKSTrajTermSynchrotronStructure = KSTrajTermSynchrotronBuilder::Attribute<string>("name") +
                                            KSTrajTermSynchrotronBuilder::Attribute<double>("enhancement") +
                                            KSTrajTermSynchrotronBuilder::Attribute<bool>("old_methode");

STATICINT sToolboxKSTrajTermSynchrotron =
    KSRootBuilder::ComplexElement<KSTrajTermSynchrotron>("kstraj_term_synchrotron");

}  // namespace katrin
