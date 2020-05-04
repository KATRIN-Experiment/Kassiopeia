#include "KSIntSurfaceSpinFlipBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

template<> KSIntSurfaceSpinFlipBuilder::~KComplexElement() {}

STATICINT sKSIntSurfaceSpinFlipStructure = KSIntSurfaceSpinFlipBuilder::Attribute<string>("name") +
                                           KSIntSurfaceSpinFlipBuilder::Attribute<double>("probability");

STATICINT sKSIntSurfaceSpinFlipElement = KSRootBuilder::ComplexElement<KSIntSurfaceSpinFlip>("ksint_surface_spin_flip");
}  // namespace katrin
