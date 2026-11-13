#include "KSGenEnergyCompositeBuilder.h"

#include "KSGenValueBoltzmannBuilder.h"
#include "KSGenValueFixBuilder.h"
#include "KSGenValueGaussBuilder.h"
#include "KSGenValueGeneralizedGaussBuilder.h"
#include "KSGenValueFermiBuilder.h"
#include "KSGenValueListBuilder.h"
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

template<> KSGenEnergyCompositeBuilder::~KComplexElement() = default;

STATICINT sKSGenEnergyCompositeStructure =
    //! the name of this energy creator.
    KSGenEnergyCompositeBuilder::Attribute<std::string>("name") +
    KSGenEnergyCompositeBuilder::Attribute<std::string>("energy") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the kinetic energy.
    KSGenEnergyCompositeBuilder::ComplexElement<KSGenValueFix>("energy_fix") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the kinetic energy.
    KSGenEnergyCompositeBuilder::ComplexElement<KSGenValueSet>("energy_set") +
    KSGenEnergyCompositeBuilder::ComplexElement<KSGenValueList>("energy_list") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the kinetic energy.
    KSGenEnergyCompositeBuilder::ComplexElement<KSGenValueUniform>("energy_uniform") +
    KSGenEnergyCompositeBuilder::ComplexElement<KSGenValueBoltzmann>("energy_boltzmann") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the kinetic energy.
    KSGenEnergyCompositeBuilder::ComplexElement<KSGenValueGauss>("energy_gauss") +
    KSGenEnergyCompositeBuilder::ComplexElement<KSGenValueGeneralizedGauss>("energy_generalized_gauss") +
    KSGenEnergyCompositeBuilder::ComplexElement<KSGenValueFermi>("energy_fermi");

STATICINT sKSGenEnergyComposite = KSRootBuilder::ComplexElement<KSGenEnergyComposite>("ksgen_energy_composite");

#ifdef Kassiopeia_USE_ROOT
STATICINT sKSGenEnergyCompositeStructureROOT =
    //! creates a (fix|set|uniform|gauss|formula) value to set the kinetic energy.
    KSGenEnergyCompositeBuilder::ComplexElement<KSGenValueFormula>("energy_formula") +
    KSGenEnergyCompositeBuilder::ComplexElement<KSGenValueHistogram>("energy_histogram");
#endif

}  // namespace katrin
