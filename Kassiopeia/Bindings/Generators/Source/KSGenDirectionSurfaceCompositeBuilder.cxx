#include "KSGenDirectionSurfaceCompositeBuilder.h"

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

template<> KSGenDirectionSurfaceCompositeBuilder::~KComplexElement() = default;

STATICINT sKSGenDirectionSurfaceCompositeStructure =
    KSGenDirectionSurfaceCompositeBuilder::Attribute<std::string>("name") +
    KSGenDirectionSurfaceCompositeBuilder::Attribute<std::string>("theta") +
    KSGenDirectionSurfaceCompositeBuilder::Attribute<std::string>("phi") +
    KSGenDirectionSurfaceCompositeBuilder::Attribute<std::string>("surfaces") +
    KSGenDirectionSurfaceCompositeBuilder::Attribute<bool>("outside") +
    KSGenDirectionSurfaceCompositeBuilder::ComplexElement<KSGenValueFix>("theta_fix") +
    KSGenDirectionSurfaceCompositeBuilder::ComplexElement<KSGenValueSet>("theta_set") +
    KSGenDirectionSurfaceCompositeBuilder::ComplexElement<KSGenValueList>("theta_list") +
    KSGenDirectionSurfaceCompositeBuilder::ComplexElement<KSGenValueUniform>("theta_uniform") +
    KSGenDirectionSurfaceCompositeBuilder::ComplexElement<KSGenValueGauss>("theta_gauss") +
    KSGenDirectionSurfaceCompositeBuilder::ComplexElement<KSGenValueGeneralizedGauss>("theta_generalized_gauss") +
    KSGenDirectionSurfaceCompositeBuilder::ComplexElement<KSGenValueAngleCosine>("theta_cosine") +
    KSGenDirectionSurfaceCompositeBuilder::ComplexElement<KSGenValueAngleSpherical>("theta_spherical") +
    KSGenDirectionSurfaceCompositeBuilder::ComplexElement<KSGenValueFix>("phi_fix") +
    KSGenDirectionSurfaceCompositeBuilder::ComplexElement<KSGenValueSet>("phi_set") +
    KSGenDirectionSurfaceCompositeBuilder::ComplexElement<KSGenValueList>("phi_list") +
    KSGenDirectionSurfaceCompositeBuilder::ComplexElement<KSGenValueUniform>("phi_uniform") +
    KSGenDirectionSurfaceCompositeBuilder::ComplexElement<KSGenValueGauss>("phi_gauss") +
    KSGenDirectionSurfaceCompositeBuilder::ComplexElement<KSGenValueGeneralizedGauss>("phi_generalized_gauss");

STATICINT sKSGenDirectionSurfaceComposite =
    KSRootBuilder::ComplexElement<KSGenDirectionSurfaceComposite>("ksgen_direction_surface_composite");


#ifdef Kassiopeia_USE_ROOT
STATICINT sKSGenDirectionSurfaceCompositeStructureROOT =
    KSGenDirectionSurfaceCompositeBuilder::ComplexElement<KSGenValueFormula>("theta_formula") +
    KSGenDirectionSurfaceCompositeBuilder::ComplexElement<KSGenValueHistogram>("theta_histogram") +
    KSGenDirectionSurfaceCompositeBuilder::ComplexElement<KSGenValueFormula>("phi_formula") +
    KSGenDirectionSurfaceCompositeBuilder::ComplexElement<KSGenValueHistogram>("phi_histogram");
#endif

}  // namespace katrin
