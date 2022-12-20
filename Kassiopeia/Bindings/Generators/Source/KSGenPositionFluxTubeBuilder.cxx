#include "KSGenPositionFluxTubeBuilder.h"

#include "KSGenGeneratorCompositeBuilder.h"
#include "KSGenValueFixBuilder.h"
#include "KSGenValueGaussBuilder.h"
#include "KSGenValueSetBuilder.h"
#include "KSGenValueUniformBuilder.h"
#include "KSRootBuilder.h"

#ifdef Kassiopeia_USE_ROOT
#include "KSGenValueFormulaBuilder.h"
#endif

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenPositionFluxTubeBuilder::~KComplexElement() = default;

STATICINT sKSGenPositionFluxTubeStructure =
    KSGenPositionFluxTubeBuilder::Attribute<std::string>("name") +
    KSGenPositionFluxTubeBuilder::Attribute<std::string>("surface") +
    KSGenPositionFluxTubeBuilder::Attribute<std::string>("space") +
    KSGenPositionFluxTubeBuilder::Attribute<std::string>("phi") +
    KSGenPositionFluxTubeBuilder::Attribute<std::string>("z") +
    KSGenPositionFluxTubeBuilder::Attribute<double>("flux") +
    KSGenPositionFluxTubeBuilder::Attribute<int>("n_integration_step") +
    KSGenPositionFluxTubeBuilder::Attribute<bool>("only_surface") +
    KSGenPositionFluxTubeBuilder::Attribute<std::string>("magnetic_field_name") +
    KSGenPositionFluxTubeBuilder::ComplexElement<KSGenValueFix>("phi_fix") +
    KSGenPositionFluxTubeBuilder::ComplexElement<KSGenValueSet>("phi_set") +
    KSGenPositionFluxTubeBuilder::ComplexElement<KSGenValueUniform>("phi_uniform") +
    KSGenPositionFluxTubeBuilder::ComplexElement<KSGenValueGauss>("phi_gauss") +
    KSGenPositionFluxTubeBuilder::ComplexElement<KSGenValueFix>("z_fix") +
    KSGenPositionFluxTubeBuilder::ComplexElement<KSGenValueSet>("z_set") +
    KSGenPositionFluxTubeBuilder::ComplexElement<KSGenValueUniform>("z_uniform") +
    KSGenPositionFluxTubeBuilder::ComplexElement<KSGenValueGauss>("z_gauss");

#ifdef Kassiopeia_USE_ROOT
STATICINT sKSGenPositionFluxTubeStructureROOT =
    KSGenPositionFluxTubeBuilder::ComplexElement<KSGenValueFormula>("r_formula") +
    KSGenPositionFluxTubeBuilder::ComplexElement<KSGenValueFormula>("z_formula");
#endif

STATICINT sToolboxKSGenPositionFluxTube =
    KSRootBuilder::ComplexElement<KSGenPositionFluxTube>("ksgen_position_flux_tube");

STATICINT sKSGenCompositePositionFluxTubeStructure =
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenPositionFluxTube>("position_flux_tube");
}  // namespace katrin
