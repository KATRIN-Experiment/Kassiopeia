#include "KSGenGeneratorCompositeBuilder.h"

#include "KSGenDirectionSphericalCompositeBuilder.h"
#include "KSGenDirectionSurfaceCompositeBuilder.h"
#include "KSGenEnergyBetaDecayBuilder.h"
#include "KSGenEnergyBetaRecoilBuilder.h"
#include "KSGenEnergyCompositeBuilder.h"
#include "KSGenEnergyKryptonEventBuilder.h"
#include "KSGenEnergyLeadEventBuilder.h"
#include "KSGenEnergyRadonEventBuilder.h"
#include "KSGenEnergyRydbergBuilder.h"
#include "KSGenLCompositeBuilder.h"
#include "KSGenLStatisticalBuilder.h"
#include "KSGenLUniformMaxNBuilder.h"
#include "KSGenMomentumRectangularCompositeBuilder.h"
#include "KSGenNCompositeBuilder.h"
#include "KSGenPositionCylindricalCompositeBuilder.h"
#include "KSGenPositionFrustrumCompositeBuilder.h"
#include "KSGenPositionMaskBuilder.h"
#include "KSGenPositionMeshSurfaceRandomBuilder.h"
#include "KSGenPositionRectangularCompositeBuilder.h"
#include "KSGenPositionSpaceRandomBuilder.h"
#include "KSGenPositionSphericalCompositeBuilder.h"
#include "KSGenPositionSurfaceAdjustmentStepBuilder.h"
#include "KSGenPositionSurfaceRandomBuilder.h"
#include "KSGenSpinCompositeBuilder.h"
#include "KSGenSpinRelativeCompositeBuilder.h"
#include "KSGenTimeCompositeBuilder.h"
#include "KSGenValueFixBuilder.h"
#include "KSGenValueGaussBuilder.h"
#include "KSGenValueListBuilder.h"
#include "KSGenValueParetoBuilder.h"
#include "KSGenValueSetBuilder.h"
#include "KSGenValueUniformBuilder.h"
#include "KSRootBuilder.h"

#ifdef Kassiopeia_USE_ROOT
#include "KSGenValueFormulaBuilder.h"
#include "KSGenValueHistogramBuilder.h"
#endif

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenGeneratorCompositeBuilder::~KComplexElement() {}

STATICINT sKSGenGeneratorCompositeStructure =
    KSGenGeneratorCompositeBuilder::Attribute<string>("name") +
    KSGenGeneratorCompositeBuilder::Attribute<string>("special") +
    KSGenGeneratorCompositeBuilder::Attribute<string>("creator") +
    KSGenGeneratorCompositeBuilder::Attribute<double>("pid") +
    KSGenGeneratorCompositeBuilder::Attribute<string>("string_id") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenEnergyComposite>("energy_composite") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenEnergyBetaDecay>("energy_beta_decay") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenEnergyBetaRecoil>("energy_beta_recoil") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenEnergyKryptonEvent>("energy_krypton_event") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenEnergyRadonEvent>("energy_radon_event") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenEnergyLeadEvent>("energy_lead_event") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenEnergyRydberg>("energy_rydberg") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenNComposite>("n_composite") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenLComposite>("l_composite") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenLUniformMaxN>("l_uniform_max_n") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenLStatistical>("l_statistical") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenPositionRectangularComposite>(
        "position_rectangular_composite") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenPositionCylindricalComposite>(
        "position_cylindrical_composite") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenPositionSphericalComposite>("position_spherical_composite") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenPositionFrustrumComposite>("position_frustrum_composite") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenPositionSpaceRandom>("position_space_random") +
    //        KSGenGeneratorCompositeBuilder::ComplexElement< KSGenPositionSurfaceRandom >( "position_surface_random" ) +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenPositionSurfaceAdjustmentStep>(
        "position_surface_adjustment_step") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenPositionMeshSurfaceRandom>("position_mesh_surface_random") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenPositionMask>("position_mask") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenMomentumRectangularComposite>(
        "momentum_rectangular_composite") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenDirectionSphericalComposite>("direction_spherical_composite") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenDirectionSurfaceComposite>("direction_surface_composite") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenSpinComposite>("spin_composite") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenSpinRelativeComposite>("spin_relative_composite") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenTimeComposite>("time_composite") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenValueFix>("pid_fix") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenValueGauss>("pid_gauss") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenValueList>("pid_list") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenValueSet>("pid_set") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenValueUniform>("pid_uniform") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenValuePareto>("pid_pareto");

STATICINT sKSGenGeneratorComposite =
    KSRootBuilder::ComplexElement<KSGenGeneratorComposite>("ksgen_generator_composite");

#ifdef Kassiopeia_USE_ROOT
STATICINT sKSGenGeneratorCompositeStructureROOT =
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenValueFormula>("pid_formula") +
    KSGenGeneratorCompositeBuilder::ComplexElement<KSGenValueHistogram>("pid_histogram");
#endif

}  // namespace katrin
