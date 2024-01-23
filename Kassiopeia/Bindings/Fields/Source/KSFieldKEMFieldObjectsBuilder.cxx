/*
 * KSFieldKEMFieldObjectsBuilder.cxx
 *
 *  Created on: 18 Jul 2016
 *      Author: wolfgang
 */

#include "KElectricQuadrupoleFieldBuilder.hh"
#include "KElectrostaticBoundaryFieldBuilder.hh"
#include "KElectrostaticConstantFieldBuilder.hh"
#include "KElectrostaticExpoFieldBuilder.hh"
#include "KElectrostaticImportFieldBuilder.hh"

#include "KInducedAzimuthalElectricFieldBuilder.hh"
#include "KMagneticDipoleFieldBuilder.hh"
#include "KMagneticSuperpositionFieldBuilder.hh"
#include "KMagnetostaticConstantFieldBuilder.hh"
#include "KMagnetostaticExpoFieldBuilder.hh"
#include "KMagnetostaticImportFieldBuilder.hh"

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
    KSRootBuilder::ComplexElement<KElectrostaticConstantField>("ksfield_electric_constant") +
    KSRootBuilder::ComplexElement<KElectrostaticExpoField>( "ksfield_electric_expo" ) +
    KSRootBuilder::ComplexElement<KElectrostaticImportField>( "ksfield_electric_import" ) +

    KSRootBuilder::ComplexElement<KElectricQuadrupoleField>("ksfield_electric_quadrupole") +
    KSRootBuilder::ComplexElement<KGElectrostaticBoundaryField>("ksfield_electrostatic") +
    KSRootBuilder::ComplexElement<KRampedElectricField>("ksfield_electric_ramped") +
    KSRootBuilder::ComplexElement<KRampedElectric2Field>("ksfield_electric_ramped_2fields") +
#ifdef Kassiopeia_USE_VTK
    KSRootBuilder::ComplexElement<KElectrostaticPotentialmap>("ksfield_electric_potentialmap") +
    KSRootBuilder::ComplexElement<KElectrostaticPotentialmapCalculator>("ksfield_electric_potentialmap_calculator") +
#endif
    KSRootBuilder::ComplexElement<KInducedAzimuthalElectricField>("ksfield_electric_induced_azi") +

    // magnetic fields
    KSRootBuilder::ComplexElement<KMagnetostaticConstantField>("ksfield_magnetic_constant") +
    KSRootBuilder::ComplexElement<KMagnetostaticExpoField>( "ksfield_magnetic_expo" ) +
    KSRootBuilder::ComplexElement<KMagnetostaticImportField>( "ksfield_magnetic_import" ) +

    KSRootBuilder::ComplexElement<KMagneticDipoleField>("ksfield_magnetic_dipole") +
    KSRootBuilder::ComplexElement<KRampedMagneticField>("ksfield_magnetic_ramped") +
    KSRootBuilder::ComplexElement<KMagneticSuperpositionField>("ksfield_magnetic_super_position") +
#ifdef Kassiopeia_USE_VTK
    KSRootBuilder::ComplexElement<KMagnetostaticFieldmap>("ksfield_magnetic_fieldmap") +
    KSRootBuilder::ComplexElement<KMagnetostaticFieldmapCalculator>("ksfield_magnetic_fieldmap_calculator") +
#endif
    KSRootBuilder::ComplexElement<KGStaticElectromagnetField>("ksfield_electromagnet");

} /* namespace katrin */
