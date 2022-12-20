#include "KSGenDirectionSphericalMagneticFieldBuilder.h"

#include "KSGenValueAngleCosineBuilder.h"
#include "KSGenValueAngleSphericalBuilder.h"
#include "KSGenValueFixBuilder.h"
#include "KSGenValueGaussBuilder.h"
#include "KSGenValueGeneralizedGaussBuilder.h"
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

template<> KSGenDirectionSphericalMagneticFieldBuilder::~KComplexElement() = default;

STATICINT sKSGenDirectionSphericalMagneticFieldStructure =
    KSGenDirectionSphericalMagneticFieldBuilder::Attribute<std::string>("name") +
    KSGenDirectionSphericalMagneticFieldBuilder::Attribute<std::string>("theta") +
    KSGenDirectionSphericalMagneticFieldBuilder::Attribute<std::string>("phi") +
    KSGenDirectionSphericalMagneticFieldBuilder::Attribute<std::string>("magnetic_field_name") +
    KSGenDirectionSphericalMagneticFieldBuilder::ComplexElement<KSGenValueFix>("theta_fix") +
    KSGenDirectionSphericalMagneticFieldBuilder::ComplexElement<KSGenValueSet>("theta_set") +
    KSGenDirectionSphericalMagneticFieldBuilder::ComplexElement<KSGenValueList>("theta_list") +
    KSGenDirectionSphericalMagneticFieldBuilder::ComplexElement<KSGenValueUniform>("theta_uniform") +
    KSGenDirectionSphericalMagneticFieldBuilder::ComplexElement<KSGenValueGauss>("theta_gauss") +
    KSGenDirectionSphericalMagneticFieldBuilder::ComplexElement<KSGenValueGeneralizedGauss>("theta_generalized_gauss") +
    KSGenDirectionSphericalMagneticFieldBuilder::ComplexElement<KSGenValueAngleCosine>("theta_cosine") +
    KSGenDirectionSphericalMagneticFieldBuilder::ComplexElement<KSGenValueAngleSpherical>("theta_spherical") +
    KSGenDirectionSphericalMagneticFieldBuilder::ComplexElement<KSGenValueFix>("phi_fix") +
    KSGenDirectionSphericalMagneticFieldBuilder::ComplexElement<KSGenValueSet>("phi_set") +
    KSGenDirectionSphericalMagneticFieldBuilder::ComplexElement<KSGenValueList>("phi_list") +
    KSGenDirectionSphericalMagneticFieldBuilder::ComplexElement<KSGenValueUniform>("phi_uniform") +
    KSGenDirectionSphericalMagneticFieldBuilder::ComplexElement<KSGenValueGauss>("phi_gauss") +
    KSGenDirectionSphericalMagneticFieldBuilder::ComplexElement<KSGenValueGeneralizedGauss>("phi_generalized_gauss");

STATICINT sKSGenDirectionSphericalMagneticField =
    KSRootBuilder::ComplexElement<KSGenDirectionSphericalMagneticField>("ksgen_direction_spherical_magnetic_field");

#ifdef Kassiopeia_USE_ROOT
STATICINT sKSGenDirectionSphericalMagneticFieldStructureROOT =
    KSGenDirectionSphericalMagneticFieldBuilder::ComplexElement<KSGenValueFormula>("theta_formula") +
    KSGenDirectionSphericalMagneticFieldBuilder::ComplexElement<KSGenValueHistogram>("theta_histogram") +
    KSGenDirectionSphericalMagneticFieldBuilder::ComplexElement<KSGenValueFormula>("phi_formula") +
    KSGenDirectionSphericalMagneticFieldBuilder::ComplexElement<KSGenValueHistogram>("phi_histogram");
#endif

}  // namespace katrin
