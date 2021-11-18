/**
 * @file KConst.h
 * @author W. Kaefer
 * @author M. Kleesiek <marco.kleesiek@kit.edu>
 * @author R. Reimann
 * @author J. Behrens <jan.behrens@kit.edu>
 * @author S. Hickford <stephanie.hickford@kit.edu>
 */

#if KConst_REFERENCE_EPOCH == 2006
#include "KConst_2006.h"

#elif KConst_REFERENCE_EPOCH == 2021

#ifndef KCONST_H_
#define KCONST_H_

#include <cmath>

/**
 * @mainpage %Kasper-Common API Reference
 *
 * Kasper's Common module contains various utility classes (IO, Math, Data Containers) and common constants.
 * Any Kasper module is by default linked against the "KCommon" module.
 */

namespace katrin
{

/**
     * This class contains various fundamental constants.
     * Values are taken from PDG edition 2021 (https://pdg.lbl.gov/2021/reviews/rpp2020-rev-phys-constants.pdf), unless pointed out otherwise.
     * The naming conventions are: normal name for SI units, a suffix _unit for something else.
     **/
namespace KConst
{

// ==================================
// ====== MATHEMATICAL NUMBERS ======
// ==================================

template<class XFloatT = double> constexpr XFloatT Pi()
{
    return 3.141592653589793238462643383279502884L;
}   //!< pi

// ===================================
// ====== FUNDAMENTAL CONSTANTS ======
// ===================================

constexpr double C()
{
    return 299792458.0;
}   //!< speed of light (c), unit: m/s, uncertainty: exact, Ref: PDG 2021

constexpr double Q()
{
    return 1.602176634E-19;
}   //!< elementary charge (e>0), unit: C, uncertainty: exact, Ref: PDG 2021

constexpr double g()
{
    return 2.00231930436;
}   //!< electron g-factor (g), unit: none (converted from (g-2)/2 value), uncertainty: 0.22 ppb, Ref: PDG 2021

constexpr double mu_B()
{
    return 9.2740100782E-24;
}   //!< Bohr magneton (mu_B), unit: A/m^2 (converted from 5.7883818060E-11 MeV/T), uncertainty: 0.3 ppb, Ref: PDG 2021

constexpr double Alpha()
{
    return 7.2973525693E-3;
}   //!< fine structure constant (alpha), unit: none, uncertainty: 0.15 ppb, Ref: PDG 2021

constexpr double Hbar()
{
    return 1.054571817E-34;
}   //!< hbar, unit: J*s, uncertainty: exact, Ref: PDG 2021

constexpr double Hbar_eV()
{
    return 6.582119569E-16;
}   //!< hbar, unit: eV*s, uncertainty: exact to calculated precision, Ref: PDG 2021

constexpr double HbarC_eV()
{
    return 197.3269804E-9;
}   //!< hbar*c, unit: eV, uncertainty: exact to given precision, Ref: PDG 2021

constexpr double kB()
{
    return 1.380649E-23;
}   //!< Boltzmann constant (k_B), unit: J/K, uncertainty: exact, Ref: PDG 2021

constexpr double kB_eV()
{
    return 8.617333262E-5;
}   //!< Boltzmann constant (k_B), unit: eV/K, uncertainty: exact to given precision, Ref: PDG 2021

constexpr double N_A()
{
    return 6.02214076E+23;
}   //!< Avogadro's constant (N_A), unit: 1/mol, uncertainty: exact, Ref: PDG 2021

constexpr double FermiConstant_eV()
{
    return 1.1663787E-5 * 1E-18 * KConst::HbarC_eV() * KConst::HbarC_eV() * KConst::HbarC_eV();
}   //!< Fermi coupling constant (G_F), unit: eV*m^3, uncertainty: 510 ppb, Ref: PDG 2021

// ===============================
// ====== ATOMIC PROPERTIES ======
// ===============================

constexpr double AtomicMassUnit_kg()
{
    return 1.66053906660E-27;
}   //!< atomic mass unit (u), unit: kg, uncertainty: 0.30 ppb, Ref: PDG 2021

constexpr double AtomicMassUnit_eV()
{
    return 931.49410242E6;
}   //!< atomic mass unit (u), unit: eV/c^2, uncertainty: 0.30 ppb, Ref: PDG 2021

constexpr double BohrRadius()
{
    return 0.529177210903E-10;
}   //!< Bohr radius (a_inf, m_prot -> infinity), unit: m, uncertainty: 0.15 ppb, Ref: PDG 2021

constexpr double BohrRadiusSquared()
{
    return 2.80028520539E-21;
}   //!< Bohr radius squared (a_inf^2), unit: m^2, uncertainty: 0.30 ppb, Ref: PDG 2021

constexpr double ERyd_eV()
{
    return 13.605693122994;
}   //!< Rydberg energy (ionization energy of atomic hydrogen for m_prot -> infinity), unit: eV, uncertainty: 0.0019 ppb = 1.9 ppt, Ref: PDG 2021

// ================================================
// ====== ELECTROMAGNETIC COUPLING CONSTANTS ======
// ================================================

constexpr double EpsNull()
{
    return 8.8541878128E-12;
}   //!< permittivity of free space (epsilon_0), unit: F/m, uncertainty: 0.15 ppb, Ref: PDG 2021

constexpr double FourPiEps()
{
    return 4. * Pi() * EpsNull();
}   //!< 4*pi*epsilon_0, unit: F/m, uncertainty: 0.15 ppb

constexpr double MuNull()
{
    return 4.E-7 * Pi() * 1.00000000055;
}   //!< permeability of free space (mu_0), unit: N/A^2, uncertainty: 0.15 ppb, Ref: PDG 2021

inline double EtaNull()
{
    return std::sqrt(MuNull() / EpsNull());
}   //!< impedance of free space (eta_0)

// ====================
// ====== MASSES ======
// ====================

constexpr double M_el_kg()
{
    return 9.1093837015E-31;
}   //!< electron mass, unit: kg, uncertainty: 0.30 ppb, Ref: PDG 2021

constexpr double M_el_eV()
{
    return 510.99895000E3;
}   //!< electron mass, unit: eV/c^2, uncertainty: 0.30 ppb, Ref: PDG 2021

constexpr double M_mu_kg()
{
    return 1.883531609e-28;
}   //!< muon mass, unit: kg (converted from 0.1134289257 u), uncertainty: 22.0 ppb, Ref: PDG 2021

constexpr double M_mu_eV()
{
    return 105.6583745E6;
}   //!< muon mass, unit: eV/c^2, uncertainty: 22.7 ppb, Ref: PDG 2021

constexpr double M_prot_kg()
{
    return 1.67262192369E-27;
}   //!< proton mass, unit: kg, uncertainty: 0.31 ppb, Ref: PDG 2021

constexpr double M_prot_eV()
{
    return 938.27208816E6;
}   //!< proton mass, unit: eV/c^2, uncertainty: 0.31 ppb, Ref: PDG 2021

constexpr double M_neut_kg()
{
    return 1.67492749793E-27;
}   //!< neutron mass, unit: kg (converted from 1.00866491595 u), uncertainty: 0.48 ppb, Ref: PDG 2021

constexpr double M_neut_eV()
{
    return 939.56542052E6;
}   //!< neutron mass, unit: eV/c^2, uncertainty: 0.57 ppb, Ref: PDG 2021

constexpr double M_deut_kg()
{
    return 3.3435837724e-27;
}   //!< deuteron mass, unit: kg (converted from 1875.61294257 MeV/c^2), uncertainty: 0.30 ppb, Ref: PDG 2021

constexpr double M_deut_eV()
{
    return 1875.61294257E6;
}   //!< deuteron mass, unit: eV/c^2, uncertainty: 0.30 ppb, Ref: PDG 2021

// =========================================================
// ====== TRITIUM PROPERTIES AND ELEMENTAL PROPERTIES ======
// =========================================================

constexpr double M_tPlus_kg()
{
    return 5.0073567446E-27;
}   //!< ionized tritium atomic mass, unit: kg, Ref: PDG 2006

constexpr double M_tPlus_eV()
{
    return 2.808920E9;
}   //!< ionized tritium atom mass, unit: eV/c^2, Ref: PDG 2006

constexpr double M_H2_kg()
{
    return 2.015650 * AtomicMassUnit_kg();
}   //!< H2 mass, unit: kg, Ref: PDG 2006

constexpr double M_H2_eV()
{
    return 2.015650 * AtomicMassUnit_eV();
}   //!< H2 mass, unit: eV/c^2, Ref: PDG 2006

constexpr double M_H2Plus_kg()
{
    return 2.01533 * AtomicMassUnit_kg();
}   //!< H2+ mass, unit: kg, Ref: NIST Chemistry WebBook, NIST Standard Reference Database Number 69, http://webbook.nist.gov/chemistry/

constexpr double M_H2Plus_eV()
{
    return 2.01533 * AtomicMassUnit_eV();
}   //!< H2+ mass, unit: eV/c^2, Ref: NIST Chemistry WebBook, NIST Standard Reference Database Number 69, http://webbook.nist.gov/chemistry/

constexpr double M_H3Plus_kg()
{
    return 3.02327 * AtomicMassUnit_kg();
}   //!< H3+ mass, unit: kg, Ref: NIST Chemistry WebBook, NIST Standard Reference Database Number 69, http://webbook.nist.gov/chemistry/

constexpr double M_H3Plus_eV()
{
    return 3.02327 * AtomicMassUnit_eV();
}   //!< H3+ mass, unit: eV/c^2, Ref: NIST Chemistry WebBook, NIST Standard Reference Database Number 69, http://webbook.nist.gov/chemistry/

constexpr double M_HMinus_kg()
{
    return 1.00849 * AtomicMassUnit_kg();
}   //!< H- mass, unit: kg, Ref: NIST Chemistry WebBook, NIST Standard Reference Database Number 69, http://webbook.nist.gov/chemistry/

constexpr double M_HMinus_eV()
{
    return 1.00849 * AtomicMassUnit_eV();
}   //!< H- mass, unit: eV/c^2, Ref: NIST Chemistry WebBook, NIST Standard Reference Database Number 69, http://webbook.nist.gov/chemistry/

constexpr double M_HePlus_kg()
{
    return 4.002053 * AtomicMassUnit_kg();
}   //!< He+ mass, unit: kg, Ref: NIST Chemistry WebBook, NIST Standard Reference Database Number 69, http://webbook.nist.gov/chemistry/

constexpr double M_HePlus_eV()
{
    return 4.002053 * AtomicMassUnit_eV();
}   //!< He+ mass, unit: eV/c^2, Ref: NIST Chemistry WebBook, NIST Standard Reference Database Number 69, http://webbook.nist.gov/chemistry/

constexpr double M_DMinus_kg()
{
    return 2.0146503577 * AtomicMassUnit_kg();
}   //!< D- mass, unit: kg, Ref: NIST Chemistry WebBook, NIST Standard Reference Database Number 69, http://webbook.nist.gov/chemistry/

constexpr double M_DMinus_eV()
{
    return 2.0146503577 * AtomicMassUnit_eV();
}   //!< D- mass, unit: eV/c^2, Ref: NIST Chemistry WebBook, NIST Standard Reference Database Number 69, http://webbook.nist.gov/chemistry/

constexpr double M_D_kg()
{
    return 2.0141017778 * AtomicMassUnit_kg();
}   //!< deuterium atomic mass, unit: kg, Ref: PDG 2006

constexpr double M_D_eV()
{
    return 2.0141017778 * AtomicMassUnit_eV();
}   //!< deuterium atomic mass, unit: eV/c^2, Ref: PDG 2006

constexpr double M_D2_kg()
{
    return 4.0282035556 * AtomicMassUnit_kg();
}   //!< deuterium molecular mass, unit: kg, Ref: PDG 2006

constexpr double M_D2_eV()
{
    return 4.0282035556 * AtomicMassUnit_eV();
}   //!< deuterium molecular mass, unit: eV/c^2, Ref: PDG 2006

constexpr double M_T_kg()
{
    return 3.0160495 * AtomicMassUnit_kg();
}   //!< tritium atomic mass, unit: kg, Ref: PDG 2006

constexpr double M_T_eV()
{
    return 3.0160495 * AtomicMassUnit_eV();
}   //!< tritium atomic mass, unit: eV/c^2, Ref: PDG 2006

constexpr double M_T2_kg()
{
    return 6.032099 * AtomicMassUnit_kg();
}   //!< tritium molecular mass, unit: kg, Ref: PDG 2006

constexpr double M_T2_eV()
{
    return 6.032099 * AtomicMassUnit_eV();
}   //!< tritium molecular mass, unit: eV/c^2, Ref: PDG 2006

constexpr double M_T2Plus_kg()
{
    return 6.0315499755 * AtomicMassUnit_kg();
}   //!< T2+ mass, unit: kg, Ref: NIST Chemistry WebBook, NIST Standard Reference Database Number 69, http://webbook.nist.gov/chemistry/

constexpr double M_T2Plus_eV()
{
    return 6.0315499755 * AtomicMassUnit_eV();
}   //!< T2+ mass, unit: eV/c^2, Ref: NIST Chemistry WebBook, NIST Standard Reference Database Number 69, http://webbook.nist.gov/chemistry/

constexpr double M_3He_kg()
{
    return 3.0160293 * AtomicMassUnit_kg();
}   //!< 3He mass, unit: kg, Ref: PDG 2006

constexpr double M_3He_eV()
{
    return 3.0160293 * AtomicMassUnit_eV();
}   //!< 3He mass, unit: eV/c^2, Ref: PDG 2006


constexpr double M_4He_kg()
{
    return 4.002603254 * AtomicMassUnit_kg();
}   //!< 4He mass, unit: kg, Ref: Atomic Weights and Isotopic Compositions with Relative Atomic Masses, NIST Standard Reference Database 144, https://www.nist.gov/pml/atomic-weights-and-isotpic-compositions-relative-atomic-masses

constexpr double M_4He_eV()
{
    return 4.002603254 * AtomicMassUnit_eV();
}   //!< 4He mass, unit: eV/c^2, Ref: Atomic Weights and Isotopic Compositions with Relative Atomic Masses, NIST Standard Reference Database 144, https://www.nist.gov/pml/atomic-weights-and-isotpic-compositions-relative-atomic-masses

constexpr double M_83Kr_kg()
{
    return 82.914127 * AtomicMassUnit_kg();
}   //!< 83Kr atomic mass, unit: kg, Ref: PDG 2006

constexpr double M_83Kr_eV()
{
    return 82.914127 * AtomicMassUnit_eV();
}   //!< 83Kr atomic mass, unit: eV/c^2, Ref: PDG 2006

constexpr double Z_He()
{
    return 2.;
}   //!< atomic number of helium, unit: none, Ref: PDG 2006

constexpr double BindingEnergy_H2()
{
    return 15.42593;
    //!< H2 binding energy, unit: eV, Ref: NIST Chemistry WebBook, http://webbook.nist.gov/cgi/cbook.cgi?ID=C1333740&Mask=20
}

constexpr double BindingEnergy_He()
{
    return 24.587387;
    //!< He binding energy, unit: eV, Ref: NIST Basic Atomic Spectroscopic Data Handbook, https://physics.nist.gov/PhysRefData/Handbook/Tables/heliumtable1.htm
}

constexpr double BindingEnergy_H2O()
{
    return 12.621;
    //!< H2O binding energy, unit: eV, Ref: NIST Chemistry WebBook, https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Units=SI&Mask=20#Ion-Energetics
}

constexpr double Viscosity_D2_30K()
{
    return 2.084E-6;
}   //!< deuterium viscosity coefficient at T = 30 K, unit: Pa*s, Ref: The Viscosity of Normal Deuterium in the Limit of Zero Density, Journal of Physical and Chemical Reference Data 16, 189 (1987)

constexpr double Viscosity_D2_80K()
{
    return 4.950E-6;
}   //!< deuterium viscosity coefficient at T = 80 K, unit: Pa*s, Ref: The Viscosity of Normal Deuterium in the Limit of Zero Density, Journal of Physical and Chemical Reference Data 16, 189 (1987)

constexpr double Viscosity_T2_30K()
{
    return 2.425E-6;
}   //!< tritium viscosity coefficient at T = 30 K, unit: Pa*s, Ref: Markus Hötzel PhD thesis, Simulation and analysis of source-related effects for KATRIN (2012) equation 5.14

constexpr double TemperatureSlipCoefficient()
{
    return 1.175;
}   //!< temperature slip coefficient, unit: none, Ref: No easy interpolation possible, from Sharipov, Tab 4, S-model "Data on the velocity slip and temperature jump coefficients [gas, mass, heat, and momentum transfer] https://doi.org/10.1109/ESIME.2004.1304046

constexpr double M_Si()
{
    return 28.0855;
}   //!< Silicon atomic mass, unit: atomic mass unit (g/mol), Ref: PDG 2021

// =============================
// ====== NEUTRINO MIXING ======
// =============================

constexpr double Deltam21sq_eV()
{
    return 7.53e-5;
}   //!< mass difference m21^2 = m2^2 - m1^2, unit: eV^2, uncertainty: 0.18e-5 eV^2, Ref: PDG 2021

constexpr double Deltam32sq_eV()
{
    return 2.453e-3;
}   //!< mass difference m32^2 = m3^2 - m2^2, unit: eV^2 (sign unknown, normal ordering), uncertainty: 0.033e-3 eV^2, Ref: PDG 2021

constexpr double InvertedDeltam32sq_eV()
{
    return 2.536e-3;
}   //!< mass difference m32^2 = m3^2 - m2^2, unit: eV^2 (sign unknown, inverted ordering), uncertainty: 0.034e-3 eV^2, Ref: PDG 2021

constexpr double Ue1sq()
{
    return 0.671;
}   //!< matrix element Ue1^2, unit: none, uncertainty: 0.0137, Ref: Calculated from 1-Ue2^2-Ue3^2

constexpr double Ue2sq()
{
    return 0.307;
}   //!< matrix element Ue2^2, unit: none, uncertainty: 0.013 Ref: PDG 2021

constexpr double Ue3sq()
{
    return 0.0220;
}   //!< matrix element Ue3^2, unit: none, uncertainty: 0.0007 Ref: PDG 2021

} /* namespace KConst */

} /* namespace katrin */

#endif  //KCONST_H

#else
#error "Unsupported value for KConst_REFERENCE_EPOCH."
#endif
