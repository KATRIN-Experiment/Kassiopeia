#include "KSIntSurfaceMultiplicationBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSIntSurfaceMultiplicationBuilder::~KComplexElement() = default;

STATICINT sKSIntSurfaceMultiplicationStructure =
    KSIntSurfaceMultiplicationBuilder::Attribute<std::string>("name") +
    KSIntSurfaceMultiplicationBuilder::Attribute<std::string>("side") +
    KSIntSurfaceMultiplicationBuilder::Attribute<double>("energy_loss_fraction") +
    KSIntSurfaceMultiplicationBuilder::Attribute<double>("required_energy_per_particle_ev");

STATICINT sKSIntSurfaceMultiplicationElement =
    KSRootBuilder::ComplexElement<KSIntSurfaceMultiplication>("ksint_surface_multiplication");
}  // namespace katrin
