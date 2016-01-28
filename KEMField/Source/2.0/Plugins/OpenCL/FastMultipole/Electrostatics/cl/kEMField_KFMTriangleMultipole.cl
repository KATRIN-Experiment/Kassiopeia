#ifndef KFMTriangleMultipole_Defined_H
#define KFMTriangleMultipole_Defined_H

#include "kEMField_defines.h"

#include "kEMField_KFMSphericalMultipoleMath.cl"
#include "kEMField_KFMRotationMatrix.cl"
#include "kEMField_KFMMultipoleRotation.cl"

#include "kEMField_GaussianCubature.cl"

CL_TYPE8
ConstructTriangleBasis(CL_TYPE4 p0, CL_TYPE4 p1, CL_TYPE4 p2)
{
    CL_TYPE8 val;

    //just in case these aren't already zero
    p0.s3 = 0.0;
    p1.s3 = 0.0;
    p2.s3 = 0.0;

    //have to construct the x, y and z-axes

    //q is closest point to p0 on line connecting p1 to p2
    CL_TYPE4 v = p2 - p1;
    CL_TYPE t = ( dot(p0, v) - dot(p1, v) )/( dot(v,v) );
    CL_TYPE4 q = p1 + v*t;

    //y-axis is aligned with the vector pointing from p1 to p2;
    CL_TYPE4 y_axis = v;
    y_axis = normalize(y_axis);

    //the line going from p0 to q is the x-axis
    CL_TYPE4 x_axis = q - p0;

    //gram-schmidt out any component of the y-axis
    CL_TYPE comp = dot(x_axis, y_axis);
    CL_TYPE4 proj = comp*y_axis;
    proj.s3 = 0.0;
    x_axis = x_axis - proj;

    CL_TYPE h = length(x_axis); //height of triangle
    x_axis = normalize(x_axis);

    //z axis from cross product
    CL_TYPE4 z_axis = cross(x_axis, y_axis);
    z_axis = normalize(z_axis);

    //now we need to find the angles from the x-axis of each side
    CL_TYPE4 temp;
    temp = p1 - p0;
    CL_TYPE PAy = dot(temp, y_axis);
    CL_TYPE PAx = dot(temp, x_axis);

    temp = p2 - p0;
    CL_TYPE PBy = dot(temp, y_axis);
    CL_TYPE PBx = dot(temp, x_axis);

    CL_TYPE phi1 = atan2(PAy, PAx);
    CL_TYPE phi2 = atan2(PBy, PBx);

    //compute the euler angles for this coordinate system
    CL_TYPE4 alpha_beta_gamma = EulerAnglesFromAxes(x_axis, y_axis, z_axis);

    val.s0 = h; //height of triangle along x-axis
    val.s1 = phi1; //lower phi angle
    val.s2 = phi2; //upper phi angle
    val.s3 = alpha_beta_gamma.s0; //alpha
    val.s4 = alpha_beta_gamma.s1; //beta
    val.s5 = alpha_beta_gamma.s2; //gamma
    val.s6 = 0.5*h*length(v); //area of triangle
    val.s7 = 0.0;

    return val;
}



void
ComputeTriangleMomentAnalyticTerms(int max_degree,
                                   __constant const CL_TYPE* equatorial_plm,
                                   CL_TYPE area,
                                   CL_TYPE h,
                                   CL_TYPE lower_angle,
                                   CL_TYPE upper_angle,
                                   CL_TYPE2* moments)
{
    I_cheb1_array(max_degree, lower_angle, upper_angle, moments); //real
    I_cheb2_array(max_degree, lower_angle, upper_angle, moments); //imag
    K_normalize_array(max_degree, h, equatorial_plm, moments);
    CL_TYPE inv_area = 1.0/area;
    int si;
    for(int l=0; l <= max_degree; l++)
    {
        for(int m=0; m <= l; m++)
        {
            si = (l*(l+1))/2 + m;
            moments[si].s0 *= inv_area;
            moments[si].s1 *= inv_area;
        }
    }
}



//returns the origin of the expansion
CL_TYPE4
TriangleMultipoleMoments(int max_degree,
                         __constant const CL_TYPE* equatorial_plm,
                         __constant const CL_TYPE* pinchon_j,
                         CL_TYPE16 vertex,
                         CL_TYPE* scratch1,
                         CL_TYPE* scratch2,
                         CL_TYPE2* moments)
{
    //read in triangle data
    CL_TYPE4 p0, p1, p2;
    p0.s0 = vertex.s0;
    p0.s1 = vertex.s1;
    p0.s2 = vertex.s2;

    p1.s0 = vertex.s4;
    p1.s1 = vertex.s5;
    p1.s2 = vertex.s6;

    p2.s0 = vertex.s8;
    p2.s1 = vertex.s9;
    p2.s2 = vertex.sA;

    //construct the coordinate basis information we need
    CL_TYPE8 basis = ConstructTriangleBasis(p0, p1, p2);

    //compute triangle moments in special coordinate system
    ComputeTriangleMomentAnalyticTerms(max_degree, equatorial_plm, basis.s6, basis.s0, basis.s1, basis.s2, moments);

    //rotate moments to cannonical basis
    ApplyFullEulerRotation_ZYZ(max_degree, pinchon_j, basis.s3, basis.s4, basis.s5, moments, scratch1, scratch2);

    //return origin of the expansion
    return p0;
}

#endif
