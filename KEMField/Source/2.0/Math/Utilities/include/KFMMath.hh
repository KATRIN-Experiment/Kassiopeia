#ifndef KFMMath_HH__
#define KFMMath_HH__



#include <cmath>
#include <limits>
#include <complex>

#include "KFMNumericalConstants.hh"
#include "KFMSquareRootUtilities.hh"


#define MAX_FACTORIAL_ARG 170
#define FACTORIAL_TABLE_SIZE 171

namespace KEMField{


/**
*
*@file KFMMath.hh
*@class KFMMath
*@brief collection of math functions used by fast multipole library
*@details
*
*Also Implements associated legendre polynomial evaluation as detailed in :
*A unified approach to the Clenshaw summation and the recursive
*computation of very high degree and order normalised associated
*Legendre functions
*S. A. Holmes, W. E. Featherstone
*Journal of Geodesy (2002) 76: 279–299
*
*The relation used is that of the standard forward column recurrance
*
*This class also contains functions used in the analytic multipole calculation
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue May 28 21:59:09 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KFMMath
{
    public:
        KFMMath(){};
        virtual ~KFMMath(){};

        //factorial of integers up to 170 (overflow after this)
        static double Factorial(int arg);

        //factorial of the sqrt of integers
        static double SqrtFactorial(int arg);

        //this is the A coefficient defined in the FFTM paper
        static double A_Coefficient(int upper, int lower);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


        static double Phi(const double* P)
        {
            return std::atan2(P[1],P[0]);
        };

        static double CosTheta(const double* P)
        {
            double r = Radius(P);
            if( std::fabs(r) < KFM_SMALLNUMBER){return 0;};
            //this check is important for ROOT multidim integrator classes, if you fail to check this
            //and the electrode you are integrating over happens to intersect the origin
            //you will get a bus error from the kADAPTIVE integrator
            return P[2]/r;
        };

        static double Radius(const double* P)
        {
            return std::sqrt(P[0]*P[0] + P[1]*P[1] + P[2]*P[2]);
        }


        static double Phi(const double* O, const double* P)
        {
            return std::atan2((P[1]-O[1]),(P[0]-O[0]));
        };

        static double CosTheta(const double* O, const double* P)
        {
            double r = Radius(O,P);
            if( std::fabs(r) < KFM_SMALLNUMBER){return 0;};
            return ((P[2]-O[2])/r);
        };

        static double Radius(const double* O, const double* P)
        {
            return std::sqrt((P[0]-O[0])*(P[0]-O[0])+(P[1]-O[1])*(P[1]-O[1])+(P[2]-O[2])*(P[2]-O[2]));
        }



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////



        //base case P_m^m(x), this should not be called directly as it is unnormalized
        //use ALP_nm instead
        static double ALP_mm(int m, const double& x);

        //returns the un-normalized associated legendre polynomials
        static double ALP_nm_unormalized(int n, int m, const double& x);

        //returns the un-normalized associated legendre polynomials
        static void ALP_nm_unormalized_array(int n_max, const double& x, double* val);

        //returns the Schmidt semi-normalized associated legendre polynomials
        static double ALP_nm(int n, int m, const double& x);

        //evaluate ALP_nm for all values of n and m from (0,0) up to and including (n_max,n_max),
        //and return results in val array, val array must have size (n_max+1)*(n_max+2)/2
        static void ALP_nm_array(int n_max, const double& x, double* val);

        //returns the derivative of the Schmidt semi-normalized associated legendre polynomials
        static double ALPDerv_nm(int n, int m, const double& x);

        //evaluate the derivative of the ALP_nm for all values of n and m from
        //(0,0) up to and including (n_max,n_max), and return results in val
        //array, val array must have size (n_max+1)*(n_max+2)/2
        static void ALPDerv_nm_array(int n_max, const double& x, double* val);


        //evaluate the all of the associated legendre polynomials and their first
        //derivatives for all values of n & m from (0,0) up to (n_max, n_max),  and returns
        // results in PlmVal, and PlmDervVal array must have size (n_max+1)*(n_max+2)/2
        static void ALPAndFirstDerv_array(int n_max, const double& x, double* PlmVal, double* PlmDervVal);



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//Collection of integrals and functions used in analytic multipole calculation

        ///uses recursion to integrate sec(x)^n
        ///power must be positive and non-zero
        ///lower and upper limits must be in the range (-pi/2, pi/2)
        static double I_secn(int n, double lower_limit, double upper_limit);

        //evaluate I_secn for all values of n from 0 up to and including n_max, and return results in val array
        //val array must have size n_max+1
        static void I_secn_array(int n_max, double lower_limit, double upper_limit, double* val);

        ///directly evalutates the integral of [sin(x)/(cos(x))^n]
        ///the power n must be positive and non-zero
        ///lower and upper limits must be in the range (-pi/2, pi/2)
        static double I_trig1(int n, double lower_limit, double upper_limit);

        //evaluate I_trig1 for all values of n from 0 up to and including n_max, and return results in val array
        //val array must have size n_max+1
        static void I_trig1_array(int n_max, double lower_limit, double upper_limit, double* val);

        ///uses recursion to evalute the integral [T_m(cos(x))/(cos(x)^(l+1))]
        ///where T_m is the m-th Chebyshev polynomial of the first kind
        ///lower and upper limits must be in the range (-pi/2, pi/2)
        ///m and l must be non-negative
        static double I_cheb1(int l, int m, double lower_limit, double upper_limit);

        //evaluate I_cheb1 for all values of l and m from (0,0) up to and including (l_max,l_max),
        //and return results in val array, val array must have size (l_max+1)*(l_max+2)/2
        static void I_cheb1_array(int l_max, double lower_limit, double upper_limit, double* val);

        //evaluate I_cheb1 for all values of l and m from (0,0) up to and including (l_max,l_max),
        //and return results in val array, val array must have size (l_max+1)*(l_max+2)/2
        //must have scratch space to store values of I_secn_array with size l_max+3
        static void I_cheb1_array_fast(int l_max, double lower_limit, double upper_limit, double* scratch, double* val);

        ///uses recursion to evalute the integral [sin(x)*U_(m-1)(cos(x))/(cos(x)^(l+1))]
        ///where U_(m-1) is the (m-1)-th Chebyshev polynomial of the second kind
        ///lower and upper limits must be in the range (-pi/2, pi/2)
        ///m and l must be non-negative
        static double I_cheb2(int l, int m, double lower_limit, double upper_limit);

        //evaluate I_cheb2 for all values of l and m from (0,0) up to and including (l_max,l_max),
        //and return results in val array, val array must have size (l_max+1)*(l_max+2)/2
        static void I_cheb2_array(int l_max, double lower_limit, double upper_limit, double* val);

        ///not an integral but a pre-factor used in the triangle
        ///multipole expansion calculation
        static double K_norm(int l, int m, double h);

       //evaluate K_norm for all values of (l,m) from 0 up to and including l_max, and return results in val array
        //val array must have size (l_max+1)*(l_max+2)/2
        static void K_norm_array(int l_max, double h, double* val);

        //evaluate K_norm for all values of (l,m) from 0 up to and including l_max, using a pre-evaluated array
        //of the associated legendre polynomials evaluated at 0 (plm) and return results in val array
        //val array must have size (l_max+1)*(l_max+2)/2
        static void K_norm_array(int l_max, double h, double* plm, double* val);


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//helper functions for computing spherical harmonics


        static void
        SphericalHarmonic_Cart(int l, int m, const double* cartesian_coords, double* result);

        static void
        RegularSolidHarmonic_Cart(int l, int m, const double* cartesian_coords, double* result);

        static void
        IrregularSolidHarmonic_Cart(int l, int m, const double* cartesian_coords, double* result);

        static std::complex<double>
        SphericalHarmonic_Cart(int l, int m, const double* cartesian_coords);

        static std::complex<double>
        SphericalHarmonic_Sph(int l, int m, const double* spherical_coords);

        static std::complex<double>
        RegularSolidHarmonic_Cart(int l, int m, const double* cartesian_coords);

        static std::complex<double>
        IrregularSolidHarmonic_Cart(int l, int m, const double* cartesian_coords);



////////////////////////////////////////////////////////////////////////////////

        static void
        RegularSolidHarmonic_Cart_Array(int n_max, const double* cartesian_coords, std::complex<double>* result);

        static void
        IrregularSolidHarmonic_Cart_Array(int n_max, const double* cartesian_coords, std::complex<double>* result);




////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

    protected:

        static const double sqrt_of_three;

        static const double factorial_table[FACTORIAL_TABLE_SIZE];

};


}//end of KEMField namespace

#endif /* KFMMath_H__ */
