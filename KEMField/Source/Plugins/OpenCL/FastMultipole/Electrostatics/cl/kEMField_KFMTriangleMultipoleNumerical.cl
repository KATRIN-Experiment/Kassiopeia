#ifndef KFMTriangleMultipoleNumerical_Defined_H
#define KFMTriangleMultipoleNumerical_Defined_H

#include "kEMField_opencl_defines.h"
#include "kEMField_KFMSphericalMultipoleMath.cl"
#include "kEMField_GaussianCubature.cl"

////////////////////////////////////////////////////////////////////////////////
//Numerical integrator implementation (version 2)

CL_TYPE4
EvaluateTriangleSurface(CL_TYPE* uv, CL_TYPE* params)
{
    //we expect the following in the parameters
    //params[0] is the degree of the current moment
    //params[1] is the order of the current moment
    //params[2] is L1
    //params[3] is L2
    //params[4] is sintheta
    //params[5] -> params[7] is n1
    //params[8] -> params[10]is n2

    CL_TYPE4 xyzj; //return value
    xyzj.s0 = params[11];
    xyzj.s1 = params[12];
    xyzj.s2 = params[13];

    //returns the point on the surface corresponding to the domain coordinates (u,v)
    //in the first three values, and the determinant of the jacobian in the last
    xyzj.s0 += uv[0]*params[5] + (1.0 - uv[0]/params[2])*uv[1]*params[8]; //x
    xyzj.s1 += uv[0]*params[6] + (1.0 - uv[0]/params[2])*uv[1]*params[9]; //y
    xyzj.s2 += uv[0]*params[7] + (1.0 - uv[0]/params[2])*uv[1]*params[10]; //z
    xyzj.s3 = sqrt( ( (1.0 - uv[0]/params[2])*params[4] )*( (1.0 - uv[0]/params[2])*params[4] ) ); //|du x dv|

    return xyzj;
}

void
EvaluateTriangleMultipoleIntegrand(CL_TYPE* params, CL_TYPE* point, CL_TYPE2* result)
{
    //we expect the following parameters to be passed to this function
    //placed in the array 'params' in the following order:
    //we expect the following in the parameters
    //params[0] is the degree of the current moment
    //params[1] is the order of the current moment
    //params[2] is L1
    //params[3] is L2
    //params[4] is sintheta
    //params[5] -> params[7] is n1
    //params[8] -> params[10]is n2

    int degree = params[0];
    int order = params[1];

    //convert point in domain (u,v)-plane to point on triangle surface
    CL_TYPE4 xyzj = EvaluateTriangleSurface(point, params);
    xyzj.s0 -= params[14];
    xyzj.s1 -= params[15];
    xyzj.s2 -= params[16];

    //now evalute phi and cos_theta for this point
    CL_TYPE radius = RadiusSingle(xyzj);
    CL_TYPE phi = PhiSingle(xyzj);
    CL_TYPE costheta = CosThetaSingle(xyzj);

    //now evaluate the solid harmonic
    CL_TYPE r_pow = pow(radius, degree);
    CL_TYPE alp = ALP_nm(degree, order, costheta);
    CL_TYPE scale = (2.0/(params[2]*params[3]*params[4]))*xyzj.s3; //inverse area * jacobian
    result[0].s0 = r_pow*alp*cos(order*phi)*scale;
    result[0].s1 = -1.0*r_pow*alp*sin(order*phi)*scale;
}

//define the integrator
GaussianCubatureComplex(EvaluateTriangleMultipoleIntegrand, 2, 1);

void
TriangleMultipoleMomentsNumerical(int max_degree,
                                  CL_TYPE4 target_origin,
                                  CL_TYPE16 vertex,
                                  __constant const CL_TYPE* abscissa,
                                  __constant const CL_TYPE* weights,
                                  CL_TYPE2* moments)
{
    //domain dimension is 2
    //range dimension is 1, but we loop over the number of moments to compute
    //p0 is treated as the origin during this evaluation

    //define needed information about the triangle
    CL_TYPE params[17];

    //params[0] is the degree of the current moment
    //params[1] is the order of the current moment
    //params[2] is L1
    //params[3] is L2
    //params[4] is sintheta
    //params[5] -> params[7] is n1
    //params[8] -> params[10] is n2
    //params[11] -> params[13] is p0
    //params[15] -> params[16] is target origin

    params[11] = vertex.s0;
    params[12] = vertex.s1;
    params[13] = vertex.s2;

    params[14] = target_origin.s0;
    params[15] = target_origin.s1;
    params[16] = target_origin.s2;

    CL_TYPE4 temp;
    temp.s0 = vertex.s4 - vertex.s0;
    temp.s1 = vertex.s5 - vertex.s1;
    temp.s2 = vertex.s6 - vertex.s2;
    temp.s3 = 0.0;

    params[2] = length(temp); //compute and set L1
    params[5] = temp.s0/params[2];
    params[6] = temp.s1/params[2];
    params[7] = temp.s2/params[2];

    temp.s0 = vertex.s8 - vertex.s0;
    temp.s1 = vertex.s9 - vertex.s1;
    temp.s2 = vertex.sA - vertex.s2;
    temp.s3 = 0.0;

    params[3] = length(temp);
    params[8] = temp.s0/params[3];
    params[9] = temp.s1/params[3];
    params[10] = temp.s2/params[3];

    params[4] = 0;
    params[4] += (params[6]*params[10] - params[7]*params[9])*(params[6]*params[10] - params[7]*params[9]);
    params[4] += (params[7]*params[8] - params[5]*params[10])*(params[7]*params[8] - params[5]*params[10]);
    params[4] += (params[5]*params[9] - params[6]*params[8])*(params[5]*params[9] - params[6]*params[8]);
    params[4] = sqrt(params[4]);

    CL_TYPE2 temp_moment[1];
    CL_TYPE lower_limits[2]; lower_limits[0] = 0.0; lower_limits[1] = 0.0;
    CL_TYPE upper_limits[2]; upper_limits[0] = params[2]; upper_limits[1] = params[3];

    //now compute the moments by gaussian cubature
    unsigned int n_eval = (max_degree+1);
    for(int l=0; l <= max_degree; l++)
    {
        params[0] = l;
        for(int m=0; m <= l; m++)
        {
            params[1] = m;
            GaussianCubatureComplex_EvaluateTriangleMultipoleIntegrand_2_1(n_eval,
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
