#ifndef KFMLineSegmentMultipoleNumerical_Defined_H
#define KFMLineSegmentMultipoleNumerical_Defined_H

#include "kEMField_opencl_defines.h"
#include "kEMField_KFMSphericalMultipoleMath.cl"
#include "kEMField_GaussianCubature.cl"

////////////////////////////////////////////////////////////////////////////////
//Numerical integrator implementation

CL_TYPE4
EvaluateLineSegment(CL_TYPE* uv, CL_TYPE* params)
{
    CL_TYPE4 xyzj; //return value

    //returns the point on the surface corresponding to the domain coordinates (u,v)
    //in the first three values, and the determinant of the jacobian in the las
    xyzj.s0 = params[3] + uv[0]*params[6]; //x
    xyzj.s1 = params[4] + uv[0]*params[7]; //y
    xyzj.s2 = params[5] + uv[0]*params[8]; //z
    xyzj.s3 = 1.0; //|du x dv|

    return xyzj;
}

void
EvaluateLineSegmentMultipoleIntegrand(CL_TYPE* params, CL_TYPE* point, CL_TYPE2* result)
{
    //we expect the following parameters to be passed to this function
    //placed in the array 'params' in the following order:
    //we expect the following in the parameters
    //params[0] is the degree of the current moment
    //params[1] is the order of the current moment
    //params[2] is L
    //params[3] -> params[5] is p0
    //params[6] -> params[8] is n1
    //params[9] -> params[11] is origin

    int degree = params[0];
    int order = params[1];

    //convert point in domain u to point on Rectangle surface
    CL_TYPE4 xyzj = EvaluateLineSegment(point, params);

    CL_TYPE4 ori;
    ori.s0 = params[9];
    ori.s1 = params[10];
    ori.s2 = params[12];
    ori.s3 = 0;

    //now evalute phi and cos_theta for this point
     CL_TYPE radius = Radius(ori, xyzj);
     CL_TYPE phi = Phi(ori, xyzj);
     CL_TYPE costheta = CosTheta(ori, xyzj);

    //now evaluate the solid harmonic
    CL_TYPE r_pow = pow(radius, degree);
    CL_TYPE alp = ALP_nm(degree, order, costheta);
    CL_TYPE scale = (1.0/params[2])*xyzj.s3; //inverse length * jacobian
    result[0].s0 = r_pow*alp*cos(order*phi)*scale;
    result[0].s1 = -1*r_pow*alp*sin(order*phi)*scale; //complex conjugate
}

//define the integrator
GaussianCubatureComplex(EvaluateLineSegmentMultipoleIntegrand, 1, 1);


void
LineSegmentMultipoleMomentsNumerical(int max_degree,
                                    CL_TYPE4 origin,
                                    CL_TYPE16 vertex,
                                    __constant const CL_TYPE* abscissa,
                                    __constant const CL_TYPE* weights,
                                    CL_TYPE2* moments)
{

    //read in the wire data
    CL_TYPE4 pA, pB, n1;

    pA.s0 = vertex.s0;
    pA.s1 = vertex.s1;
    pA.s2 = vertex.s2;
    pA.s3 = 0.0;

    pB.s0 = vertex.s4;
    pB.s1 = vertex.s5;
    pB.s2 = vertex.s6;
    pB.s3 = 0.0;

    CL_TYPE len = distance(pA, pB);
    CL_TYPE inv_len = 1.0/len;
    n1 = inv_len*(pB - pA);

    CL_TYPE params[12];

    params[2] = len;

    params[3] = pA.s0;
    params[4] = pA.s1;
    params[5] = pA.s2;

    params[6] = n1.s0;
    params[7] = n1.s1;
    params[8] = n1.s2;

    params[9] = origin.s0;
    params[10] = origin.s1;
    params[11] = origin.s2;


    CL_TYPE2 temp_moment[1];
    CL_TYPE lower_limits[1]; lower_limits[0] = 0.0;
    CL_TYPE upper_limits[1]; upper_limits[0] = len;

    //now compute the moments by gaussian cubature
    unsigned int n_eval = (max_degree+1);
    for(int l=0; l <= max_degree; l++)
    {
        params[0] = l;
        for(int m=0; m <= l; m++)
        {
            params[1] = m;
            GaussianCubatureComplex_EvaluateLineSegmentMultipoleIntegrand_1_1(n_eval,
                                                                           abscissa,
                                                                           weights,
                                                                           lower_limits,
                                                                           upper_limits,
                                                                           params,
                                                                           temp_moment);

            moments[(l*(l+1))/2 + m] = temp_moment[0];
        }
    }
}


#endif
