#ifndef KFMSphericalMultipoleExpansionEvaluation_Defined_H
#define KFMSphericalMultipoleExpansionEvaluation_Defined_H

#include "kEMField_opencl_defines.h"
#include "kEMField_KFMSphericalMultipoleMath.cl"

void
ComputeRadialAndAzimuthalWeights(int Degree, CL_TYPE radius, CL_TYPE phi,
                                 CL_TYPE* RPower, CL_TYPE* CosMPhi, CL_TYPE* SinMPhi)
{
    //intial values needed for recursion to compute cos(m*phi) and sin(m*phi) arrays
    CL_TYPE sin_phi = sin(phi);
    CL_TYPE sin_phi_over_two = sin(phi/2.0);
    CL_TYPE eta_real = -2.0*sin_phi_over_two*sin_phi_over_two;
    CL_TYPE eta_imag = sin_phi;
    CosMPhi[0] = 1.0;
    SinMPhi[0] = 0.0;
    //scratch space space
    CL_TYPE a, b, mag2, delta;
    //intial value need for recursion on powers of radius
    RPower[0] = 1.0;

    for(int j = 1; j <= Degree; j++)
    {
        //compute needed power of radius
        RPower[j] = radius*RPower[j-1];

        //compute needed value of cos(m*phi) and sin(m*phi) (see FFT class for this method)
        a = CosMPhi[j-1] + eta_real*CosMPhi[j-1] - eta_imag*SinMPhi[j-1];
        b = SinMPhi[j-1] + eta_imag*CosMPhi[j-1] + eta_real*SinMPhi[j-1];
        mag2 = a*a + b*b;
        delta = rsqrt(mag2);
        CosMPhi[j] = a*delta;
        SinMPhi[j] = b*delta;
    }
}



CL_TYPE
ElectricPotential(int Degree, CL_TYPE* RPower, CL_TYPE* CosMPhi, CL_TYPE* SinMPhi,
                  CL_TYPE* PlmVal, CL_TYPE* RealMoments, CL_TYPE* ImagMoments)
{
    CL_TYPE potential = 0.0;
    CL_TYPE partial_sum = 0.0;

    int si0, si;
    for(int j = 0; j <= Degree; j++)
    {
        si0 = (j*(j+1))/2;
        partial_sum = 0.0;
        for(int k = 1; k <= j; k++)
        {
            si = si0 + k;
            partial_sum += 2.0*(CosMPhi[k]*RealMoments[si] - SinMPhi[k]*ImagMoments[si])*PlmVal[si];
        }
        partial_sum += RealMoments[si0]*PlmVal[si0];
        potential += RPower[j]*partial_sum;
    }

    return potential;
}



CL_TYPE4
ElectricField(int Degree, CL_TYPE cos_theta, CL_TYPE sin_theta, CL_TYPE* RPower, CL_TYPE* CosMPhi,
              CL_TYPE* SinMPhi, CL_TYPE* PlmVal, CL_TYPE* PlmDervVal, CL_TYPE* RealMoments, CL_TYPE* ImagMoments)
{

    //TODO provide alternate version of this function when evaluation takes
    //place near the z-pole

    CL_TYPE dr = 0.0; //derivative w.r.t. to radius
    CL_TYPE dt = 0.0; //(1/r)*(derivative w.r.t. to theta)
    CL_TYPE dp = 0.0; //(1/(r*sin(theta)))*(derivative w.r.r. to phi)

    CL_TYPE inverse_sin_theta = 1.0/sin_theta;
    CL_TYPE re_product;
    CL_TYPE im_product;
    CL_TYPE partial_sum_dr = 0.0;
    CL_TYPE partial_sum_dt = 0.0;
    CL_TYPE partial_sum_dp = 0.0;

    int si0, si;
    for(int j = 1; j <= Degree; j++)
    {
        si0 = (j*(j+1))/2;
        partial_sum_dr = 0.0;
        partial_sum_dt = 0.0;
        partial_sum_dp = 0.0;

        for(int k = 1; k <= j; k++)
        {
            si = si0 + k;
            re_product = 2.0*(CosMPhi[k]*RealMoments[si] - SinMPhi[k]*ImagMoments[si]);
            im_product = 2.0*(CosMPhi[k]*ImagMoments[si] + SinMPhi[k]*RealMoments[si]);
            partial_sum_dr += re_product*PlmVal[si];
            partial_sum_dt += re_product*PlmDervVal[si];
            partial_sum_dp += k*im_product*PlmVal[si];
        }

        partial_sum_dr += RealMoments[si0]*PlmVal[si0];
        partial_sum_dt += RealMoments[si0]*PlmDervVal[si0];
        dr += j*partial_sum_dr*RPower[j-1];
        dt += partial_sum_dt*RPower[j-1];
        dp -= inverse_sin_theta*partial_sum_dp*RPower[j-1];
    }


    //set field components in spherical coordinates
    CL_TYPE SphField[3];
    SphField[0] = dr;
    SphField[1] = dt;
    SphField[2] = dp;

    //now we must define the matrix to transform
    //the field from spherical to cartesian coordinates
    CL_TYPE XForm[3][3];
    XForm[0][0] = sin_theta*CosMPhi[1];
    XForm[0][1] = cos_theta*CosMPhi[1];
    XForm[0][2] = -1.0*SinMPhi[1];
    XForm[1][0] = sin_theta*SinMPhi[1];
    XForm[1][1] = cos_theta*SinMPhi[1];
    XForm[1][2] = CosMPhi[1];
    XForm[2][0] = cos_theta;
    XForm[2][1] = -1.0*sin_theta;
    XForm[2][2] = 0.0);

    CL_TYPE CartField[3];
    CartField[0] = 0.0;
    CartField[1] = 0.0;
    CartField[2] = 0.0;

    //apply transformation
    for(int i=0; i<3; i++)
    {
        for(int j=0; j<3; j++)
        {
            CartField[i] += XForm[i][j] * SphField[j];
        }
    }

    //return the field values
    CL_TYPE4 ret_val;

    ret_val.s0 = -1.0*CartField[0];
    ret_val.s1 = -1.0*CartField[1];
    ret_val.s2 = -1.0*CartField[2];
    ret_val.s3 = 0.0;

    return ret_val;
}



#endif
