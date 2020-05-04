#include "KSIntSpinFlipBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

STATICINT sKSIntSpinFlip = KSRootBuilder::ComplexElement<KSIntSpinFlip>("ksint_spin_flip");

STATICINT sKSIntSpinFlipStructure = KSIntSpinFlipBuilder::Attribute<string>("name");

}  // namespace katrin
