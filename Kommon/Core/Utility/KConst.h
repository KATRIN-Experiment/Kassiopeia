#ifndef KCONST_H_
#define KCONST_H_

#include <cmath>

/**
 * @mainpage %Kasper-Common API Reference
 *
 * Kasper's Common module contains various utility classes (IO, Math, Data Containers) and common constants.
 * Any Kasper module is by default linked against the Kasper-Common library.
 */

/**
 @file KConst.h
 @brief Contains katrin::KConst
 @details
 *
 */

namespace katrin {

/**
 * \author W. Kaefer
 * \brief  this class contains various fundamental constants.
 *  \details Values are taken from PDG edition 2006, unless pointed out otherwise. The naming conventions are: normal name for SI units, a suffix _unit for something else.
 * For example, masses are usually give in both kg and eV with no suffix in the former case and suffix _eV in the latter.
 * Masses will probably be moved to some sort of particle table at some point.
 **/
class KConst
{
    public:
        KConst();
        virtual ~KConst();

        //mathematical numbers
        static inline double Pi()
        {
            return 3.14159265358979311599796346854;
        } //!< pi

        static inline double C()
        {
            return 299792458.;
        } //!< c im m/s

        static inline double Q()
        {
            return 1.60217653E-19;
        } //!< elementary charge  in C(>0)

        static inline double Alpha()
        {
            return 7.297352568E-3;
        } //!< fine structure constant alpha

        static inline double Hbar()
        {
            return 1.05457168E-34;
        }//!< hbar in J s-1

        static inline double HbarC_eV()
        {
            return 197.326968E-9;
        }//!<hbar c in m eV.

        static inline double kB()
        {
            return 1.3806505E-23;
        }//!< Boltzmann constant J/K

        static inline double kB_eV()
        {
            return 8.617343E-5;
        }//!< Boltzmann constant eV/K

        static inline double N_A()
        {
            return 6.022141E+23;
        }//!< Avogadro Constant in 1/mol

        //atomic properties
        static inline double AtomicMassUnit()
        {
            return 1.66053886E-27;
        } //!< unified atomic mass unit in kg

        static inline double AtomicMassUnit_eV()
        {
            return 931.49404E6;
        } //!< unified atomic mass unit in eV/c^2

        static inline double BohrRadius()
        {
            return 0.5291772108E-10;
        } //!<Bohr radius (M_prot -> infinity)

        static inline double BohrRadiusSquared()
        {
            return 2.8002852043e-21;
        } //!<squared Bohr radius

        static inline double ERyd_eV()
        {
            return 13.6056923;
        } //!< Rydberg energy (ionization energy of atomic hydrogen for m_prot -> infinity)

        //EM coupling constants
        static inline double EpsNull()
        {
            return 8.854187817E-12;
        } //!< epsilon0, Constant of Newtons force.

        static inline double FourPiEps()
        {
            return 4. * Pi() * EpsNull();
        } //!< 4  pi  epsilon0, Constant of Newtons force.

        static inline double MuNull()
        {
            return 4.E-7 * Pi();
        }//!< permeability of free space

        static inline double EtaNull()
        {
            return sqrt(MuNull() /EpsNull());
        }//!< impedance of free space

        //masses
        static inline double M_el()
        {
            return 9.1093826E-31;
        } //!< electron mass in kg

        static inline double M_el_eV()
        {
            return 510.998918E3;
        } //!< electron mass in ev

        static inline double M_mu()
        {
            return 1.88353160e-28;
        } //!< muon mass in kg

        static inline double M_mu_eV()
        {
            return 105.6583692E6;
        } //!< muon mass in ev

        static inline double M_prot()
        {
            return 1.67262171E-27;
        } //!< proton mass in kg

        static inline double M_prot_eV()
        {
            return 938.272029E6;
        } //!< proton mass in ev

        static inline double M_neut()
        {
            return 1.674927464E-27;
        } //!< neutron mass in kg

        static inline double M_neut_eV()
        {
            return 939.565360E6;
        } //!< neutron mass in ev

        static inline double M_deut()
        {
            return 3.34358334e-27;
        } //!< deuteron mass in kg

        static inline double M_deut_eV()
        {
            return 1875.61282E6;
        } //!< deuteron mass in eV.

        //Tritium properties
        static inline double M_T2()
        {
            return 6 * AtomicMassUnit();
        } //!< tritium molecule mass in kg (estimation. needs a literature search)

        static inline double M_T2_eV()
        {
            return 6 * AtomicMassUnit_eV();
        } //!< tritium molecule mass in eV/c^2

        static inline double Viscosity()
        {
            return 2.425E-6;
        } //!< tritium viscosity coefficient at T=30K [Pa s] (cite? Sharipov?)

        //Silicon properties
        static inline double M_Si()
        {
            return 28.086;
        } //!< Silicon atomic mass in g per mol

        static inline double FermiConstant_eV()
        {
            return 1.16637E-5 * 1E-18 * KConst::HbarC_eV() * KConst::HbarC_eV() * KConst::HbarC_eV();
        } //!< Fermi coupling constant [eVm^3]

        //some SSC specific stuff
        static inline double costhetaC()
        {
            return 0.9750; //!< cos(Cabibbo angle). Reference?
        }

        static inline double MatrixM()
        {
            return 2.348; //!< nuclear matrix element. Reference?
        }

        static inline double gV()
        {
            return 1.0; //!< alternative to nuclear matrix element: use (g_V^2 + 3 * g_A^2) Reference?
        }

        static inline double gA()
        {
            return 1.247; //!< alternative to nuclear matrix element: use (g_V^2 + 3 * g_A^2) Reference?
        }

        //neutrino mixing
        static inline double Deltam21sq_eV()
        {
        	return 7.5e-5; // m2^2 - m1^2; Unit is eV^2; Reference: PDG 6/18/2012
        }

        static inline double Deltam32sq_eV()
        {
        	return 2.32e-3; //m3^2 - m2^2; Unit is eV^2; sign unknown; Reference: PDG 6/18/2012
        }

        static inline double Ue1sq()
        {
        	return 0.672; // calculated from the angles in the reference; Reference: PDG 6/18/2012
        }

        static inline double Ue2sq()
        {
        	return 0.303; // calculated from the angles in the reference; Reference: PDG 6/18/2012
        }

        static inline double Ue3sq()
        {
        	return 0.025; // calculated from the angles in the reference; Reference: PDG 6/18/2012
        }
};

}

#endif //KCONST_H
