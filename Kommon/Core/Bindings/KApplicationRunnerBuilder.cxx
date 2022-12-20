#include "KApplicationRunnerBuilder.h"

#include "KRoot.h"

using namespace std;

namespace katrin
{

STATICINT sKApplicationRunnerHook = KRootBuilder::ComplexElement<KApplicationRunner>("run");

STATICINT sKApplicationRunnerStructure = KApplicationRunnerBuilder::Attribute<std::string>("name") +
                                         KApplicationRunnerBuilder::ComplexElement<KNamedReference>("app");

}  // namespace katrin
