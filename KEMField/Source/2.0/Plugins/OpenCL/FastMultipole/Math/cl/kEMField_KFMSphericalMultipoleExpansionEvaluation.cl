#ifndef KFMSphericalMultipoleExpansionEvaluation_Defined_H
#define KFMSphericalMultipoleExpansionEvaluation_Defined_H

#include "kEMField_opencl_defines.h"
#include "kEMField_KFMSphericalMultipoleMath.cl"

CL_TYPE
ElectricPotential(int Degree, CL_TYPE4 sph_coord, CL_TYPE2* moments)
{
    //the variable sph_coord stores: (r, cos_theta, phi, sin_theta)

    CL_TYPE potential = 0.0;
    CL_TYPE partial_sum = 0.0;

    CL_TYPE r, x, phi;
    r = sph_coord.s0; //radius
    x = sph_coord.s1; //cos(theta)
    phi = sph_coord.s2; //phi

    int si0, si;
    for(int j = 0; j <= Degree; j++)
    {
        si0 = j*(j+1)/2;
        partial_sum = 0.0;
        for(int k = 1; k <= j; k++)
        {
            si = si0 + k;
            partial_sum += 2.0*(cos(k*phi)*moments[si].s0 - sin(k*phi)*moments[si].s1)*ALP_nm(j,k,x);
        }
        partial_sum += moments[si0].s0*ALP_nm(j,0,x);
        potential += pown(r, j)*partial_sum;
    }

    return potential;
}



CL_TYPE4
ElectricField(int Degree, CL_TYPE4 sph_coord, CL_TYPE2* moments)
{
    //TODO provide alternate version of this function when evaluation takes
    //place near the z-pole

    //the variable sph_coord stores: (r, cos_theta, phi, sin_theta)
    CL_TYPE r, ct, st, phi;
    r = sph_coord.s0; //radius
    ct = sph_coord.s1; //cos(theta)
    st = sph_coord.s3; //sin(theta)
    phi = sph_coord.s2; //phi
    CL_TYPE inverse_sin_theta = 1.0/st; // 1.0/sin(theta)

    CL_TYPE dr = 0.0; //derivative w.r.t. to radius
    CL_TYPE dt = 0.0; //(1/r)*(derivative w.r.t. to theta)
    CL_TYPE dp = 0.0; //(1/(r*sin(theta)))*(derivative w.r.r. to phi)

    CL_TYPE re_product;
    CL_TYPE im_product;
    CL_TYPE partial_sum_dr = 0.0;
    CL_TYPE partial_sum_dt = 0.0;
    CL_TYPE partial_sum_dp = 0.0;

    int si0, si;
    CL_TYPE cp, sp, plm, rpow;
    for(int j = 1; j <= Degree; j++)
    {
        si0 = j*(j+1)/2;
        partial_sum_dr = 0.0;
        partial_sum_dt = 0.0;
        partial_sum_dp = 0.0;

        for(int k = 1; k <= j; k++)
        {
            si = si0 + k;
            cp = cos(k*phi);
            sp = sin(k*phi);
            plm = ALP_nm(j,k,ct);
            re_product = 2.0*(cp*moments[si].s0 - sp*moments[si].s1);
            im_product = 2.0*(cp*moments[si].s1 + sp*moments[si].s0);
            partial_sum_dr += re_product*plm;
            partial_sum_dt += re_product*ALPDerv_nm(j,k,ct);
            partial_sum_dp += k*im_product*plm;
        }

        partial_sum_dr += moments[si0].s0*ALP_nm(j,0,ct);
        partial_sum_dt += moments[si0].s0*ALPDerv_nm(j,0,ct);

        rpow = pown(r, j-1);
        dr += j*partial_sum_dr*rpow;
        dt += partial_sum_dt*rpow;
        dp -= inverse_sin_theta*partial_sum_dp*rpow;
    }


    //set field components in spherical coordinates
    CL_TYPE SphField[3];
    SphField[0] = dr;
    SphField[1] = dt;
    SphField[2] = dp;

    //now we must define the matrix to transform
    //the field from spherical to cartesian coordinates
    cp = cos(phi);
    sp = sin(phi);

    CL_TYPE XForm[3][3];
    XForm[0][0] = st*cp;
    XForm[0][1] = ct*cp;
    XForm[0][2] = -1.0*sp;
    XForm[1][0] = st*sp;
    XForm[1][1] = ct*sp;
    XForm[1][2] = cp;
    XForm[2][0] = ct;
    XForm[2][1] = -1.0*st;
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
