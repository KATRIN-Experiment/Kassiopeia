#ifndef KFMMultipoleRotation_Defined_H
#define KFMMultipoleRotation_Defined_H

#include "kEMField_opencl_defines.h"

//______________________________________________________________________________


//applies a rotation about the z-axis of the degree l moments
//stored in in_mom, and returns them in out_mom (executed in real basis)
void
ApplyZRotMatrix(int l, CL_TYPE angle, CL_TYPE* in_mom, CL_TYPE* out_mom)
{
    CL_TYPE u, v, c, s, a, b, mag2, delta;
    int psi, nsi;

    //m=0 is a direct copy
    out_mom[l*(l+1)] = in_mom[l*(l+1)];

    //the following recursive method for computing cos(mx) and sin(mx) is from
    //An Algorithm for Computing the Mixed Radix Fast Fourier Transform
    //Richard C. Singleton
    //IEEE Transactions on Audio and Electroacoustics Vol. Au-17 No. 2 June 1969

    //intial values needed for recursion
    CL_TYPE sin_theta = sin(angle);
    CL_TYPE sin_theta_over_two = sin(angle/2.0);
    CL_TYPE eta_real = -2.0*sin_theta_over_two*sin_theta_over_two;
    CL_TYPE eta_imag = sin_theta;

    //initial values for recursion (m=0)
    c = 1.0;
    s = 0.0;

    //only m>=1 terms are changed by rotation
    for(int m=1; m <= l; m++)
    {
        a = c + eta_real*c - eta_imag*s;
        b = s + eta_imag*c + eta_real*s;

        //re-scale to correct round off errors
        mag2 = a*a + b*b;
        delta = 1.0/sqrt(mag2);

        a *= delta;
        b *= delta;

        c = a;
        s = b;

        psi = l*(l+1)+m;
        nsi = l*(l+1)-m;

        u = in_mom[nsi];
        v = in_mom[psi];

        //perform rotation
        out_mom[nsi] = c*u + s*v;
        out_mom[psi] = c*v - s*u;


    }
}

//______________________________________________________________________________

//applies the pinchon-j matrix to the vector of degree l moments
//stored in in_mom, and returns them in out_mom (executed in real basis)
void
ApplyJMatrix(int l, __constant const CL_TYPE* jmat, CL_TYPE* in_mom, CL_TYPE* out_mom)
{
    //expects matrix to be of size 2l + 1 by 2l+1
    int size = 2*l+1;
    CL_TYPE sum;

    int offset = l*l;
    for(int row=0; row < size; row++)
    {
        sum = 0.0;

        for(int col=0; col < size; col++)
        {
            sum  += (jmat[col + row*size])*in_mom[offset + col];
        }

        out_mom[offset + row] = sum;
    }
}

//______________________________________________________________________________

void
ConvertSphericalHarmonicsComplexToRealBasis(int max_degree, CL_TYPE2* complex_moments, CL_TYPE* real_moments)
{
    //convert complex spherical harmonics into real spherical harmonics basis
    int rb_psi, rb_nsi, cb_si;

    //rb_psi = real basis, positive m storage index
    //rb_nsi = real basis, negative m storage index
    //cb_si = complex basis, (only positive m) storage index

    CL_TYPE fac;
    for(int l=0; l <= max_degree; l++)
    {
        //m=0 moment have no imaginary part
        cb_si = l*(l+1)/2;
        rb_psi = l*(l+1);

        real_moments[rb_psi] = sqrt((2.0*l + 1.0)/(4.0*M_PI))*complex_moments[cb_si].s0;
        fac = sqrt((2.0*l + 1.0)/(2.0*M_PI));

        for(int m=1; m<=l; m++)
        {
            cb_si = l*(l+1)/2 + m;
            rb_psi = l*(l+1) + m;
            rb_nsi = rb_psi - 2*m;
            real_moments[rb_psi] = fac*complex_moments[cb_si].s0;
            real_moments[rb_nsi] = fac*complex_moments[cb_si].s1;
        }
    }
}

//______________________________________________________________________________

void
ConvertSphericalHarmonicsRealToComplexBasis(int max_degree, CL_TYPE* real_moments, CL_TYPE2* complex_moments)
{
    //convert real spherical harmonics to the complex spherical harmonic basis

    int rb_psi, rb_nsi, cb_si;

    //rb_psi = real basis, positive m storage index
    //rb_nsi = real basis, negative m storage index
    //cb_si = complex basis, (only positive m) storage index

    CL_TYPE fac;
    for(int l=0; l <= max_degree; l++)
    {
        //m=0 moment have no imaginary part
        cb_si = l*(l+1)/2;
        rb_psi = l*(l+1);
        complex_moments[cb_si].s0 = sqrt((4.0*M_PI)/(2.0*l + 1.0))*real_moments[rb_psi];
        complex_moments[cb_si].s1 = 0.0;

        fac = sqrt((2.0*M_PI)/(2.0*l + 1.0));

        for(int m=1; m<=l; m++)
        {
            cb_si = l*(l+1)/2 + m;
            rb_psi = l*(l+1) + m;
            rb_nsi = rb_psi - 2*m;
            complex_moments[cb_si].s0 = fac*real_moments[rb_psi];
            complex_moments[cb_si].s1 = fac*real_moments[rb_nsi];
        }
    }

}


//______________________________________________________________________________
void
ApplyFullEulerRotation_ZYZ(int max_degree,
                      __constant const CL_TYPE* pinchon_j,
                      CL_TYPE alpha,
                      CL_TYPE beta,
                      CL_TYPE gamma,
                      CL_TYPE2* complex_moments,
                      CL_TYPE* real_moments_scratch1,
                      CL_TYPE* real_moments_scratch2)
{

    //first convert the complex moments to the real basis
    ConvertSphericalHarmonicsComplexToRealBasis(max_degree, complex_moments, real_moments_scratch1);

    //apply the rotation in the real basis
    int j_mat_index = 0;
    for(int l=0; l <= max_degree; l++)
    {
        ApplyZRotMatrix(l, alpha, real_moments_scratch1, real_moments_scratch2); //alpha
        ApplyJMatrix(l, &(pinchon_j[j_mat_index]), real_moments_scratch2, real_moments_scratch1);
        ApplyZRotMatrix(l, beta, real_moments_scratch1, real_moments_scratch2); //beta
        ApplyJMatrix(l, &(pinchon_j[j_mat_index]), real_moments_scratch2, real_moments_scratch1 );
        ApplyZRotMatrix(l, gamma, real_moments_scratch1, real_moments_scratch2); //gamma
        j_mat_index += (2*l+1)*(2*l+1);
    }

    //convert the real moments back to the complex basis
    ConvertSphericalHarmonicsRealToComplexBasis(max_degree, real_moments_scratch2, complex_moments);
}

//______________________________________________________________________________

void
ApplySingleRotation_Z(int max_degree,
                         CL_TYPE alpha,
                         CL_TYPE2* complex_moments,
                         CL_TYPE* real_moments_scratch1,
                         CL_TYPE* real_moments_scratch2)
{

    //first convert the complex moments to the real basis
    ConvertSphericalHarmonicsComplexToRealBasis(max_degree, complex_moments, real_moments_scratch1);

    //apply the rotation in the real basis
    for(int l=0; l <= max_degree; l++)
    {
        ApplyZRotMatrix(l, alpha, real_moments_scratch1, real_moments_scratch2); //alpha
    }

    //convert the real moments back to the complex basis
    ConvertSphericalHarmonicsRealToComplexBasis(max_degree, real_moments_scratch2, complex_moments);

}

#endif /* KFMMultipoleRotation_Defined_H */
