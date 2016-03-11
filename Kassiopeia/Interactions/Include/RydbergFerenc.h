//
// Created by trost on 26.05.15.
//

#ifndef KASPER_RYDBERGFERENC_H
#define KASPER_RYDBERGFERENC_H

#define nMAX 8000

#include "KField.h"
/*
  Test computations:

    1. Comparison of RadInt2Gordon with RadInt2Num; the latter computes the
   integral of RadialH(n,l,r)*RadialH(np,lp,r)*r*r*r  (eq. 63.1, page 262 in the Bethe-Salpeter book)
   by numerical integration. See the files HydrogenAtomNumint.cc and QuadGaussLegendre.cc.
   Note: for very small values of RadInt2Gordon ( below 10^-15) , these two values disagree. Nevertheless, further
   checks (see below) show that the numerical integration values RadInt2Num are wrong in these cases, and the
   RadInt2Gordon values are reliable also in these special cases.
   In these cases the transition rates are extremely small, i.e. they can be taken as zero.

   2. Comparison with eqs. 63.4, 63.5,  61.7 and table 13 (pages 262-264, 257) in the Bethe-Salpeter book.

   3. Comparison with eqs. 68 and 66 in:
         D. Dewangan, Phys. Reports 511 (2012) 1.

   4. Comparison with eq. 21 in Dewangan 2012 paper, and eqs. 2.23-2.27 in:
         P. Storey, D. Hummer: Comp. Phys. Comm. 66 (1991) 129,
         and eqs. 52-53 in:
         J. D. Hey, J. Phys. B 39 (2006) 2641,
         and eq. C2 in:
         J. Watson, J. Phys. B 39 (2006) 1889.


   5. Comparison with table 1 in:
      D. Hoang-Binh, Comp. Phys. Commun. 166 (2005) 191.

   6.  Comparison with the semiclassical spontaneous decay lifetime formula in the abstract
         and with the lifetime values of table 1 in:
         H. Marxer, L. Spruch, Phys. Rev. A 43 (1991) 1268

   7. Comparison with the spontaneous decay and BBR induced emission and absorption
       approximative formulas of Glukhov et al., J. Phys. B 43 (2010) 125002.

*/


namespace Kassiopeia {

    class FBBRionization {
        public:
            FBBRionization(double aT, int an, int al);
            virtual ~FBBRionization();

        public:
            double operator() (double E);

        private:
            /** \brief Photoionization cross section of a general (n,l) state of hydrogen atom.
             *  n, l: principal and angular momentum quantum numbers of the initial discrete state.
             * omega: photon energy in atomic units; omega=1 corresponds to  E=27.211 eV.
             * omega has to be larger than the ionization energy of the (n,l) state;
             * if omega is smaller, then the function returns zero.
             * The cross section is also in atomic units:  sigma=1 corresponds to a0^2 (Bohr radius squared).
             * Eqs. 1 and 2 from Burgess 1965 paper are used, in modified form.
             * Agrees with eq. 23 of Glukhov 2010.
             */
            double SigmaPhotoionization(int n,int l,double E);

            /** \brief Square of integral of RadialH(n,l,r)*RadialH(E,lp)*r*r*r from zero to infinity.
             * We use the paper : A. Burgess: Tables of hydrogenic photoionization cross-sections
             * and recombination coefficients,
             * Memoirs of the Royal Astronomical Society 69 (1965) 1,
             * http://adsabs.harvard.edu/abs/1965MmRAS..69....1B
             *  n, l: principal and angular momentum quantum numbers of the initial discrete state.
             * lp=l+sign: angular momentum quantum number of the final continuous state; sign has to be +1 or -1.
             * E: outgoing (free) electron energy in atomic units; E=1 corresponds to 27.2 eV
             *  E has to be positive.
             * If lp<0 :  returns 0.  !!!
             * Maximal value of n: nMAX (set by #define in a separate file)
             */
            double RadInt2BoundFreeBurgess(int n, int l, double E, int sign);


        K_SET_GET(double, T)
        K_SET_GET(int, n)
        K_SET_GET(int, l)

        private:
            double LogN[2 * nMAX + 10];
            double LogNFactor[2 * nMAX + 10];
            double logpi;
            double Ehigh, Clow, Chigh;
            double fC;
            double fAtomic_C;
            double fAtomic_kB;
    };


    class RydbergCalculator {

    public:
        RydbergCalculator();
        virtual ~RydbergCalculator();

    private:
        double LogN[2 * nMAX + 10];
        double LogNFactor[2 * nMAX + 10];
        double Ehigh, Clow, Chigh;
        double fAtomicTimeUnit;
        double fAtomic_C;
        double fAtomic_kB;
        FBBRionization* fFBBRIon;

    public:
        /**
        * \brief This function computes the hypergeometric function by using the recurrence relation of
        * D. Hoang-Binh, Comp. Phys. Commun. 166 (2005) 191.
        * We use b-recurrence.
        * a and b should be non-positive integers, and c  positive (non-zero) integer.
        *
        * \param a
        * \param b
        * \param c
        * \param x
        * \param E
        */
        double HypergeometricFHoangBinh(int a, int b, int c, double x, double &E);

        /**
         * \brief Square of integral of RadialH(n,l,r)*RadialH(np,lp,r)*r*r*r from zero to infinity.
         +   n,np,  l, lp=l+sign: principal and angular momentum quantum numbers.
         +   lp=l+sign; sign has to be +1 or -1
         +   If lp<0 or lp>np-1:  returns 0.  !!!
         +   We use eq. 63.2 (page 262) in Bethe-Salpeter:Quantum mechanics of one and two electron atoms,
         +   which is for sign=-1.
         +   This is known in the literature by Gordon formula (published by W. Gordon in 1929).
         +   Integral RadialH(n,l,r)*RadialH(np,l+sign,r) =Integral RadialH(np,l+sign,r)*RadialH(n,l,r),
         +   therefore for sign=+1 we change: np--> n, n--> np,  l+1 --> l
         +   Maximal values of n and np: nMAX
         *
         *  \param n
         *  \param l
         *  \param np
         *  \param sign
         *
         *  \return Rate
         *
         */
        double RadInt2Gordon(int n, int l, int np, int sign);


        /**
         * \brief Rate of spontaneous emission from state (n,l) to state (np,l+sign) in s^-1 .
         * sign has to be either -1 or +1.
         * We use atomic units here.
         *
         * \param n
         * \param l
         * \param np
         * \param sign
         */
        double Psp(int n, int l, int np, int sign);

        /**
         * \brief Rate of spontaneous emission from state (n,l) to all np<n states  with lp=l+-1 in s^-1
         *
         * \param n
         * \param l
         */
        double Pspsum(int n, int l);

        /**
         * \brief Rate of BBR induced transition from state (n,l) to state (np,l+sign) in s^-1 .
         * sign has to be either -1 or +1.
         * n and np have to be different integers!
         * We use atomic units here.
         *
         * \param T double Temperature in Kelvin
         * \param n
         * \param l
         * \param np
         * \param sign
         */
        double PBBR(double T, int n, int l, int np, int sign);

        /**
         * \brief BBR induced decay (stimulated emission) rates: we sum for all np from 1 to n-1 and all possible lp
         * \param T
         * \param n
         * \param l
         */
        double PBBRdecay(double T, int n, int l);

        /**
         * \brief BBR induced excitation (photon absorption) rates: we sum for all np from n+1 to npmax and all possible lp
         * \param T
         * \param n
         * \param l
         * \param npmax
         */
        double PBBRexcitation(double T, int n, int l, int npmax);

        /**
         * \brief This function computes the total spontaneous emission rate Psptotal from
         * the initial state (n,l),
         * and generates the  quantum numbers np, lp of the final state, using the
         * inverse transform sampling method for discrete distribution.
         * Discrete distribution function: PDF[i], i=1,...,2*N.
         * Cumulative distribution function: CDF[i], i=1,...,2*N+1.
         * N=npmax-npmin+1.
         *
         * \param n
         * \param l
         * \param Psptotal
         * \param np
         * \param lp
         */
        void SpontaneousEmissionGenerator(int n, int l, double &Psptotal, int &np, int &lp);

        /**
         * \brief This function computes the total BBR induced transition rate PBBRtotal from
         * the initial state (n,l),
         * and generates the  quantum numbers np, lp of the final state, using the
         * inverse transform sampling method for discrete distribution.
         * T: temperature in K.
         * Discrete distribution function: PDF[i], i=1,...,2*N.
         * Cumulative distribution function: CDF[i], i=1,...,2*N+1.
         * N=npmax-npmin+1.
         *
         * \param n
         * \param l
         * \param PBBRtotal
         * \param np
         * \param lp
         */
        void BBRTransitionGenerator(double T, int n, int l, double &PBBRtotal, int &np, int &lp);

        /** \brief BBR induced ionization rate of the (n,l) state in s^-1.
         * n, l: principal and angular momentum quantum numbers of the Rydberg state.
         * T: temperature in K.
         * The numerical integration is performed over the electron energy E (atomic units: a.u.)
         *  from zero to infinity.
         * This integration region is divided into smaller intervals, and the integration is done by
         * Gauss-Legendre within each subinterval;  see function QuadGaussLegendreH.
         * step1factor: the first interval step of the integration is from zero to step1,
         *  where step1=Enl*step1factor, and Enl is the ionization energy of the (n,l) state (a.u.).
         * Recommended value: step1factor=1.
         * tol: tolerance number; the integration procedure stops if the ratio of the integral value of
         * the last interval and the total integral is smaller then tol.
         * Recommended value: tol=1.e-6.
         * Ninteg: number of function evaluations within 1 integratio step.
         * Recommended value: Ninteg=16.
         * The integration result is accurate if it is not sensitive to the step1factor, tol and Ninteg values.
         */
        double PBBRionization(double T, int n, int l,double step1factor,double tol,int Ninteg);

    };



}
#endif //KASPER_RYDBERGFERENC_H

