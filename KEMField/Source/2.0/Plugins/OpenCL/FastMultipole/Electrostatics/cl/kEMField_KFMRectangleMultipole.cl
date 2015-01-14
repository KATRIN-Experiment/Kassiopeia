#ifndef KFMRectangleMultipole_Defined_H
#define KFMRectangleMultipole_Defined_H


#include "kEMField_defines.h"

#include "kEMField_KFMSphericalMultipoleMath.cl"
#include "kEMField_KFMMultipoleRotation.cl"
#include "kEMField_KFMTriangleMultipole.cl"


CL_TYPE4
RectangleMultipoleMoments(int max_degree,
                         __constant const CL_TYPE* equatorial_plm,
                         __constant const CL_TYPE* pinchon_j,
                         CL_TYPE16 vertex,
                         CL_TYPE* scratch1,
                         CL_TYPE* scratch2,
                         CL_TYPE2* moments_scratch,
                         CL_TYPE2* moments_output)
{
    int size = (max_degree+1)*(max_degree+2)/2;

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

    CL_TYPE4 del_a = p[a_mid] - p[0];
    CL_TYPE4 del_b = p[b_mid] - p[0];
    CL_TYPE4 centroid = p[0] + 0.5*del_a + 0.5*del_b;

    //triangle A consists of the centroid, p[0] and p[a_mid];
    //compute moments of triangle A
    //construct the coordinate basis information we need
    CL_TYPE8 basis = ConstructTriangleBasis(centroid, p[0], p[a_mid]);
    CL_TYPE alpha = basis.s3;
    CL_TYPE beta = basis.s4;
    CL_TYPE gamma = basis.s5;
    ComputeTriangleMomentAnalyticTerms(max_degree, equatorial_plm, 2.0*basis.s6, basis.s0, basis.s1, basis.s2, moments_output);

    //triangle B consists of the centroid, p[0] and p[b_mid];
    //compute moments of triangle B
    //construct the coordinate basis information we need
    basis = ConstructTriangleBasis(centroid, p[0], p[b_mid]);
    ComputeTriangleMomentAnalyticTerms(max_degree, equatorial_plm, 2.0*basis.s6, basis.s0, basis.s1, basis.s2, moments_scratch);

    //rotate the moments of triangle B by 90 degrees
    ApplySingleRotation_Z(max_degree, M_PI/2.0, moments_scratch, scratch1, scratch2);

    //now we add the two triangle's moments together
    for(int i=0; i<size; i++)
    {
        //add the moments of both triangles together
        moments_output[i] += moments_scratch[i];
        //copy output back into scratch
        moments_scratch[i] = moments_output[i];
    }

    //compute the moments of the mirror image by rotation
    ApplySingleRotation_Z(max_degree, M_PI, moments_scratch, scratch1, scratch2);

    //add in the mirror image moments to the moments of triangle A and B
    for(int i=0; i<size; i++)
    {
        moments_output[i] += moments_scratch[i];
    }

    //now apply full euler rotation to moments
    ApplyFullEulerRotation_ZYZ(max_degree, pinchon_j, alpha, beta, gamma, moments_output, scratch1, scratch2);

    //rescale all moments by 1/2
    for(int i=0; i<size; i++)
    {
        moments_output[i] *=  0.5;
    }

    return centroid; //return expansion origin
}

#endif
