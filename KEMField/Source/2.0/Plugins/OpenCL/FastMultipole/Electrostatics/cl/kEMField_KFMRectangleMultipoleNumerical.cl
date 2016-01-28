#ifndef KFMRectangleMultipoleNumerical_Defined_H
#define KFMRectangleMultipoleNumerical_Defined_H

#include "kEMField_defines.h"
#include "kEMField_KFMSphericalMultipoleMath.cl"
#include "kEMField_GaussianCubature.cl"

////////////////////////////////////////////////////////////////////////////////
//Numerical integrator implementation

CL_TYPE4
EvaluateRectangleSurface(CL_TYPE* uv, CL_TYPE* params)
{
    //we expect the following in the parameters
    //params[0] is the degree of the current moment
    //params[1] is the order of the current moment
    //params[2] is L1
    //params[3] is L2
    //params[4] is unused
    //params[5] -> params[7] is n1
    //params[8] -> params[10] is n2
    //params[11] -> params[13] is origin
    //params[14] -> params[16] is p0


    CL_TYPE4 xyzj; //return value

    //returns the point on the surface corresponding to the domain coordinates (u,v)
    //in the first three values, and the determinant of the jacobian in the las

    xyzj.s0 = params[14] + uv[0]*params[5] + uv[1]*params[8]; //x
    xyzj.s1 = params[15] + uv[0]*params[6] + uv[1]*params[9]; //y
    xyzj.s2 = params[16] + uv[0]*params[7] + uv[1]*params[10]; //z
    xyzj.s3 = 1.0; //|du x dv|

    return xyzj;
}

void
EvaluateRectangleMultipoleIntegrand(CL_TYPE* params, CL_TYPE* point, CL_TYPE2* result)
{
    //we expect the following parameters to be passed to this function
    //placed in the array 'params' in the following order:
    //we expect the following in the parameters
    //params[0] is the degree of the current moment
    //params[1] is the order of the current moment
    //params[2] is L1
    //params[3] is L2
    //params[4] is unused
    //params[5] -> params[7] is n1
    //params[8] -> params[10] is n2
    //params[11] -> params[13] is origin
    //params[14] -> params[16] is p0


    int degree = params[0];
    int order = params[1];

    //convert point in domain (u,v)-plane to point on Rectangle surface
    CL_TYPE4 xyzj = EvaluateRectangleSurface(point, params);

    CL_TYPE4 ori;
    ori.s0 = params[11];
    ori.s1 = params[12];
    ori.s2 = params[13];
    ori.s3 = 0;

    //now evalute phi and cos_theta for this point
     CL_TYPE radius = Radius(ori, xyzj);
     CL_TYPE phi = Phi(ori, xyzj);
     CL_TYPE costheta = CosTheta(ori, xyzj);

    //now evaluate the solid harmonic
    CL_TYPE r_pow = pow(radius, degree);
    CL_TYPE alp = ALP_nm(degree, order, costheta);
    CL_TYPE scale = (1.0/(params[2]*params[3]))*xyzj.s3; //inverse area * jacobian
    result[0].s0 = r_pow*alp*cos(order*phi)*scale;
    result[0].s1 = -1*r_pow*alp*sin(order*phi)*scale;
}

//define the integrator
GaussianCubatureComplex(EvaluateRectangleMultipoleIntegrand, 2, 1);

void
RectangleMultipoleMomentsNumerical(int max_degree,
                                  CL_TYPE4 origin,
                                  CL_TYPE16 vertex,
                                  __constant const CL_TYPE* abscissa,
                                  __constant const CL_TYPE* weights,
                                  CL_TYPE2* moments)
{
    //domain dimension is 2
    //range dimension is 1, but we loop over the number of moments to compute
    //p0 is treated as the origin during this evaluation

    //vertices of the rectangle
    CL_TYPE4 p[4];

    p[0].s0 = vertex.s0;
    p[0].s1 = vertex.s1;
    p[0].s2 = vertex.s2;
    p[0].s3 = 0.0;

    p[1].s0 = vertex.s4;
    p[1].s1 = vertex.s5;
    p[1].s2 = vertex.s6;
    p[1].s3 = 0.0;

    p[2].s0 = vertex.s8;
    p[2].s1 = vertex.s9;
    p[2].s2 = vertex.sA;
    p[2].s3 = 0.0;

    p[3].s0 = vertex.sC;
    p[3].s1 = vertex.sD;
    p[3].s2 = vertex.sE;
    p[3].s3 = 0.0;

    //we have to split the rectangle into four triangles
    //here we figure out which sets of points we need to use in each triangle
    CL_TYPE d01 = distance(p[0],p[1]);
    CL_TYPE d02 = distance(p[0],p[2]);
    CL_TYPE d03 = distance(p[0],p[3]);

    int opposite_corner_index, a_mid, b_mid;
    CL_TYPE max_dist = d01; opposite_corner_index = 1; a_mid = 2; b_mid = 3;
    if(d02 > max_dist){max_dist = d02; opposite_corner_index = 2; a_mid = 3; b_mid = 1;}
    if(d03 > max_dist){max_dist = d03; opposite_corner_index = 3; a_mid = 1; b_mid = 2;}

    //define needed information about the Rectangle
    CL_TYPE params[17];

    //params[0] is the degree of the current moment
    //params[1] is the order of the current moment
    //params[2] is L1
    //params[3] is L2
    //params[4] is unused -> TODO revise this parameter list
    //params[5] -> params[7] is n1
    //params[8] -> params[10]is n2
    CL_TYPE4 temp;
    temp = p[a_mid] - p[0];

    params[2] = length(temp); //compute and set L1
    params[5] = temp.s0/params[2];
    params[6] = temp.s1/params[2];
    params[7] = temp.s2/params[2];

    temp = p[b_mid] - p[0];

    params[3] = length(temp);
    params[8] = temp.s0/params[3];
    params[9] = temp.s1/params[3];
    params[10] = temp.s2/params[3];


    params[11] = origin.s0;
    params[12] = origin.s1;
    params[13] = origin.s2;

    params[14] = vertex.s0;
    params[15] = vertex.s1;
    params[16] = vertex.s2;

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
            GaussianCubatureComplex_EvaluateRectangleMultipoleIntegrand_2_1(n_eval,
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
