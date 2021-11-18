/**
 * @file KConst.h
 * @author W. Kaefer
 * @author M. Kleesiek <marco.kleesiek@kit.edu>
 */

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
     * Values are taken from PDG edition 2006, unless pointed out otherwise. The naming conventions are: normal name for SI units, a suffix _unit for something else.
     **/
namespace KConst
{
//mathematical numbers
template<class XFloatT = double> constexpr XFloatT Pi()
{
    return 3.141592653589793238462643383279502884L;
}  //!< pi

constexpr double C()
{
    return 299792458.0;
}  //!< c im m/s

constexpr double Q()
{
    return 1.60217653E-19;
}  //!< elementary charge  in C(>0)

constexpr double g()
{
    return 2.002319304361;
}  //!< electron g-factor (>0)

constexpr double g_N()
{
    return 9.80665;
}  //!< standard gravitational acceleration (m/s^2)

constexpr double mu_B()
{
    return 9.27400968E-24;
}  //!< Bohr magneton in J/T

constexpr double Alpha()
{
    return 7.2973525664E-3;
}  //!< fine structure constant alpha

constexpr double Hbar()
{
    return 1.05457168E-34;
}  //!< hbar in J*s

constexpr double Hbar_eV()
{
    return Hbar() / Q();
}  //!< hbar in eV*s

constexpr double HbarC_eV()
{
    return 197.326968E-9;
}  //!< hbar c in m eV.

constexpr double kB()
{
    return 1.3806505E-23;
}  //!< Boltzmann constant J/K

constexpr double kB_eV()
{
    return 8.617343E-5;
}  //!< Boltzmann constant eV/K

constexpr double N_A()
{
    return 6.022141E+23;
}  //!< Avogadro Constant in 1/mol

//atomic properties
constexpr double AtomicMassUnit_kg()
{
    return 1.66053886E-27;
}  //!< unified atomic mass unit in kg

constexpr double AtomicMassUnit_eV()
{
    return 931.49404E6;
}  //!< unified atomic mass unit in eV/c^2

constexpr double BohrRadius()
{
    return 0.5291772108E-10;
}  //!< Bohr radius (M_prot -> infinity)

constexpr double BohrRadiusSquared()
{
    return 2.8002852043e-21;
}  //!< squared Bohr radius

constexpr double ERyd_eV()
{
    return 13.6056923;
}  //!< Rydberg energy (ionization energy of atomic hydrogen for m_prot -> infinity)

//EM coupling constants
constexpr double EpsNull()
{
    return 8.854187817E-12;
}  //!< epsilon0, Constant of Newtons force.

constexpr double FourPiEps()
{
    return 4. * Pi() * EpsNull();
}  //!< 4  pi  epsilon0, Constant of Newtons force.

constexpr double MuNull()
{
    return 4.E-7 * Pi();
}  //!< permeability of free space

inline double EtaNull()
{
    return std::sqrt(MuNull() / EpsNull());
}  //!< impedance of free space

//masses
constexpr double M_el_kg()
{
    return 9.1093826E-31;
}  //!< electron mass in kg

constexpr double M_el_eV()
{
    return 510.998918E3;
}  //!< electron mass in ev/c^2

constexpr double M_mu_kg()
{
    return 1.88353160e-28;
}  //!< muon mass in kg

constexpr double M_mu_eV()
{
    return 105.6583692E6;
}  //!< muon mass in ev/c^2

constexpr double M_prot_kg()
{
    return 1.67262171E-27;
}  //!< proton mass in kg

constexpr double M_prot_eV()
{
    return 938.272029E6;
}  //!< proton mass in ev/c^2

constexpr double M_neut_kg()
{
    return 1.674927464E-27;
}  //!< neutron mass in kg

constexpr double M_neut_eV()
{
    return 939.565360E6;
}  //!< neutron mass in ev/c^2

constexpr double M_deut_kg()
{
    return 3.34358334e-27;
}  //!< deuteron mass in kg

constexpr double M_deut_eV()
{
    return 1875.61282E6;
}  //!< deuteron mass in eV/c^2

//Tritium properties
constexpr double M_tPlus_kg()
{
    return 5.00735626e-27;
}  //!ionized tritium atom mass in kg

constexpr double M_tPlus_eV()
{
    return 2.808920E9;
}  //!ionized tritium atom mass in eV/c^2

constexpr double M_H2_kg()
{
    return 2.015650 * AtomicMassUnit_kg();
}

constexpr double M_H2_eV()
{
    return 2.015650 * AtomicMassUnit_eV();
}

//Mass values for H2+, H3+, H-, D-, T2+, and He+ from NIST Chemistry WebBook, NIST Standard Reference Database Number 69, http://webbook.nist.gov/chemistry/

constexpr double M_H2Plus_kg()
{
    return 2.01533 * AtomicMassUnit_kg();
}  //!< H2+ mass in kg

constexpr double M_H2Plus_eV()
{
    return 2.01533 * AtomicMassUnit_eV();
}

constexpr double M_H3Plus_kg()
{
    return 3.02327 * AtomicMassUnit_kg();
}  //!< H3+ mass in kg

constexpr double M_H3Plus_eV()
{
    return 3.02327 * AtomicMassUnit_eV();
}

constexpr double M_HMinus_kg()
{
    return 1.00849 * AtomicMassUnit_kg();
}  //!< H- mass in kg

constexpr double M_HMinus_eV()
{
    return 1.00849 * AtomicMassUnit_eV();
}

constexpr double M_HePlus_kg()
{
    return 4.002053 * AtomicMassUnit_kg();
}  //!< He+ mass in kg

constexpr double M_HePlus_eV()
{
    return 4.002053 * AtomicMassUnit_eV();
}

constexpr double M_DMinus_kg()
{
    return 2.0146503577 * AtomicMassUnit_kg();
}  //!< D- mass in kg

constexpr double M_DMinus_eV()
{
    return 2.0146503577 * AtomicMassUnit_eV();
}

constexpr double M_D_kg()
{
    return 2.0141017778 * AtomicMassUnit_kg();
}  //!< deuterium atom mass in kg

constexpr double M_D_eV()
{
    return 2.0141017778 * AtomicMassUnit_eV();
}  //!< deuterium atom mass in eV/c^2

constexpr double M_D2_kg()
{
    return 4.0282035556 * AtomicMassUnit_kg();
}  //!< deuterium molecule mass in kg

constexpr double M_D2_eV()
{
    return 4.0282035556 * AtomicMassUnit_eV();
}  //!< deuterium molecule mass in eV/c^2

constexpr double M_T_kg()
{
    return 3.0160495 * AtomicMassUnit_kg();
}  //!< tritium atom mass in kg

constexpr double M_T_eV()
{
    return 3.0160495 * AtomicMassUnit_eV();
}  //!< tritium atom mass in eV/c^2

constexpr double M_T2_kg()
{
    return 6.032099 * AtomicMassUnit_kg();
}  //!< tritium molecule mass in kg

constexpr double M_T2_eV()
{
    return 6.032099 * AtomicMassUnit_eV();
}  //!< tritium molecule mass in eV/c^2

constexpr double M_T2Plus_kg()
{
    return 6.0315499755 * AtomicMassUnit_kg();
}  //!< T2+ mass in kg

constexpr double M_T2Plus_eV()
{
    return 6.0315499755 * AtomicMassUnit_eV();
}

constexpr double M_3He_kg()
{
    return 3.0160293 * AtomicMassUnit_kg();
}

constexpr double M_3He_eV()
{
    return 3.01602932243 * AtomicMassUnit_eV();
}

//Value for 4He mass taken from Atomic Weights and Isotopic Compositions with Relative Atomic Masses, NIST Standard Reference Database 144
//https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses
/*constexpr double M_4He_kg()
          {
          return 4.002603254 * AtomicMassUnit_kg();
          } //!< 4He mass in kg

          constexpr double M_4He_eV()
          {
          return 4.002603254 * AtomicMassUnit_eV();
          } //!< 4He mass in eV/c^2
          */

constexpr double Z_He()
{  //atomic number of Helium
    return 2.;
}

constexpr double BindingEnergy_H2()
{
    return 15.42593;  //eV
    //Value from NIST Chemistry WebBook: http://webbook.nist.gov/cgi/cbook.cgi?ID=C1333740&Mask=20
}

constexpr double BindingEnergy_He()
{
    return 24.587387;  //eV
    //Value from NIST Basic Atomic Spectroscopic Data Handbook: http://physics.nist.gov/PhysRefData/Handbook/Tables/heliumtable1.htm
}

constexpr double BindingEnergy_H2O()
{
    return 12.621;  //eV
    //Value from NIST Chemistry WebBook: https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Units=SI&Mask=20#Ion-Energetics
}

constexpr double Viscosity_T2_30K()
{
    return 2.425E-6;
}  //!< tritium viscosity coefficient at T=30K [Pa s] (cite? Sharipov?)

constexpr double Viscosity_D2_30K()
{
    return 2.084E-6;
}  //!< deuterium viscosity coefficient at T=30K [Pa s] (https://doi.org/10.1063/1.555778, The Viscosity of Normal Deuterium in the Limit of Zero Density, Journal of Physical and Chemical Reference Data 16, 189 (1987)

/**
         * temperature slip coefficient, no easy interpolation possible. from Sharipov, Tab 4, S-model
         * "Data on the velocity slip and temperature jump coefficients [gas, mass, heat and momentum transfer]"
         * https://doi.org/10.1109/ESIME.2004.1304046
         */
constexpr double TemperatureSlipCoefficient()
{
    return 1.175;
}

//Silicon properties
constexpr double M_Si()
{
    return 28.086;
}  //!< Silicon atomic mass in g per mol

constexpr double M_83Kr_kg()
{
    return 82.914127 * AtomicMassUnit_kg();
}  //!< 83Krypton atomic mass in kg

constexpr double M_83Kr_eV()
{
    return 82.914127 * AtomicMassUnit_eV();
}  //!< 83Krypton atomic mass in eV/c^2

constexpr double FermiConstant_eV()
{
    return 1.16637E-5 * 1E-18 * KConst::HbarC_eV() * KConst::HbarC_eV() * KConst::HbarC_eV();
}  //!< Fermi coupling constant [eVm^3]

//neutrino mixing
constexpr double Deltam21sq_eV()
{
    return 7.5e-5;  // m2^2 - m1^2; Unit is eV^2; Reference: PDG 6/18/2012
}

constexpr double Deltam32sq_eV()
{
    return 2.32e-3;  //m3^2 - m2^2; Unit is eV^2; sign unknown; Reference: PDG 6/18/2012
}

constexpr double InvertedDeltam32sq_eV()
{
    return 2.536e-3;
}   //!< mass difference m32^2 = m3^2 - m2^2, unit: eV^2 (sign unknown, inverted ordering), uncertainty: 0.034e-3 eV^2, Ref: PDG 2021

constexpr double Ue1sq()
{
    return 0.672;  // calculated from the angles in the reference; Reference: PDG 6/18/2012
}

constexpr double Ue2sq()
{
    return 0.303;  // calculated from the angles in the reference; Reference: PDG 6/18/2012
}

constexpr double Ue3sq()
{
    return 0.025;  // calculated from the angles in the reference; Reference: PDG 6/18/2012
}

} /* namespace KConst */

} /* namespace katrin */

#endif  //KCONST_H
