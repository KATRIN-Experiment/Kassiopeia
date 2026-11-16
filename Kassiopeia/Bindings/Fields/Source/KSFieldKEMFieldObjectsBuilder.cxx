/*
 * KSFieldKEMFieldObjectsBuilder.cxx
 *
 *  Created on: 18 Jul 2016
 *      Author: wolfgang
 */

#include "KElectricQuadrupoleFieldBuilder.hh"
#include "KElectrostaticBoundaryFieldBuilder.hh"
#include "KElectrostaticConstantFieldBuilder.hh"
#include "KInducedAzimuthalElectricFieldBuilder.hh"
#include "KMagneticDipoleFieldBuilder.hh"
#include "KMagneticSuperpositionFieldBuilder.hh"
#include "KMagnetostaticConstantFieldBuilder.hh"
#include "KRampedElectric2FieldBuilder.hh"
#include "KRampedElectricFieldBuilder.hh"
#include "KRampedMagneticFieldBuilder.hh"
#include "KSRootBuilder.h"
#include "KStaticElectromagnetFieldBuilder.hh"

#ifdef Kassiopeia_USE_VTK
#include "KElectrostaticPotentialmapBuilder.hh"
#include "KMagnetostaticFieldmapBuilder.hh"
#endif

using namespace KEMField;
using namespace Kassiopeia;

namespace katrin
{

STATICINT sKSRootConstant =
    // electric fields
    //! @deprecated define fields in the <kemfield> section instead (legacy alias)
    KSRootBuilder::ComplexElement<KElectrostaticConstantField>("ksfield_electric_constant") +
    //! @deprecated define fields in the <kemfield> section instead (legacy alias)
    KSRootBuilder::ComplexElement<KElectricQuadrupoleField>("ksfield_electric_quadrupole") +
    //! @deprecated define fields in the <kemfield> section instead (legacy alias)
    KSRootBuilder::ComplexElement<KGElectrostaticBoundaryField>("ksfield_electrostatic") +
    //! @deprecated define fields in the <kemfield> section instead (legacy alias)
    KSRootBuilder::ComplexElement<KRampedElectricField>("ksfield_electric_ramped") +
    //! @deprecated define fields in the <kemfield> section instead (legacy alias)
    KSRootBuilder::ComplexElement<KRampedElectric2Field>("ksfield_electric_ramped_2fields") +
#ifdef Kassiopeia_USE_VTK
    //! @deprecated define fields in the <kemfield> section instead (legacy alias)
    KSRootBuilder::ComplexElement<KElectrostaticPotentialmap>("ksfield_electric_potentialmap") +
    //! @deprecated define fields in the <kemfield> section instead (legacy alias)
    KSRootBuilder::ComplexElement<KElectrostaticPotentialmapCalculator>("ksfield_electric_potentialmap_calculator") +
#endif
    //! @deprecated define fields in the <kemfield> section instead (legacy alias)
    KSRootBuilder::ComplexElement<KInducedAzimuthalElectricField>("ksfield_electric_induced_azi") +

    // magnetic fields
    //! @deprecated define fields in the <kemfield> section instead (legacy alias)
    KSRootBuilder::ComplexElement<KMagnetostaticConstantField>("ksfield_magnetic_constant") +
    //! @deprecated define fields in the <kemfield> section instead (legacy alias)
    KSRootBuilder::ComplexElement<KMagneticDipoleField>("ksfield_magnetic_dipole") +
    //! @deprecated define fields in the <kemfield> section instead (legacy alias)
    KSRootBuilder::ComplexElement<KRampedMagneticField>("ksfield_magnetic_ramped") +
    //! @deprecated define fields in the <kemfield> section instead (legacy alias)
    KSRootBuilder::ComplexElement<KMagneticSuperpositionField>("ksfield_magnetic_super_position") +
#ifdef Kassiopeia_USE_VTK
    //! @deprecated define fields in the <kemfield> section instead (legacy alias)
    KSRootBuilder::ComplexElement<KMagnetostaticFieldmap>("ksfield_magnetic_fieldmap") +
    //! @deprecated define fields in the <kemfield> section instead (legacy alias)
    KSRootBuilder::ComplexElement<KMagnetostaticFieldmapCalculator>("ksfield_magnetic_fieldmap_calculator") +
#endif
    //! @deprecated define fields in the <kemfield> section instead (legacy alias)
    KSRootBuilder::ComplexElement<KGStaticElectromagnetField>("ksfield_electromagnet");

} /* namespace katrin */
