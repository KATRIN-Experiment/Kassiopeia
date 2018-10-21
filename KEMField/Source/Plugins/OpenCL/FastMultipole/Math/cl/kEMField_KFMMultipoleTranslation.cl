#ifndef KFMMultipoleTranslator_Defined_H
#define KFMMultipoleTranslator_Defined_H

#include "kEMField_opencl_defines.h"

#include "kEMField_KFMSphericalMultipoleMath.cl"
#include "kEMField_KFMMultipoleRotation.cl"
#include "kEMField_KFMRotationMatrix.cl"

CL_TYPE
Factorial(int n)
{
    if(n <= 0){return 1.0;};

    CL_TYPE result = 1.0;

    for(int i=n; i > 0; i--)
    {
        result *= i;
    }

    return result;
}

CL_TYPE
A_Factor(int upper, int lower)
{
    CL_TYPE result = 1.0;
    if( (lower & 1) != 0)
    {
        result = -1.0;
    }

    result *= rsqrt( Factorial( abs(lower-upper) )*Factorial( abs(lower+upper) ) );

    return result;
}


int
IsPhysical(int j, int k, int n, int m)
{
    if( (abs(k) <= j) && (abs(m) <= n) &&
        (n <= j) && (abs(k - m) <= (j - n)) )
    {
        return 1;
    }
    else
    {
        return 0;
    }
}


CL_TYPE
NormalizationCoeff(int j, int k, int n, int m)
{
    CL_TYPE pre = 1.0;


    if( (k-m)*m < 0)
    {
        int power = min(abs(k-m), abs(m));
        if( (power & 1) != 0)
        {
            pre *= -1.0;
        }

    }

    return pre*(A_Factor(m,n)*A_Factor(k-m,j-n))/(A_Factor(k,j));
}

CL_TYPE2
ResponseFunctionM2M(CL_TYPE4 del, int j, int k, int n, int m)
{
    CL_TYPE2 result;
    CL_TYPE radial_factor = pow(del.s0, n);
    CL_TYPE plm = ALP_nm(n, abs(m), del.s1);
    CL_TYPE cosmphi = cos(-1.0*m*del.s2);
    CL_TYPE sinmphi = sin(-1.0*m*del.s2);
    CL_TYPE norm = NormalizationCoeff(j,k,n,m);

    result.s0 = norm*radial_factor*cosmphi*plm;
    result.s1 = norm*radial_factor*sinmphi*plm;

    return result;
}

//______________________________________________________________________________


void
TranslateMultipoleMoments(int max_degree,
                              __constant const CL_TYPE* a_coefficient,
                              CL_TYPE4 source_origin,
                              CL_TYPE4 target_origin,
                              CL_TYPE2* source_moments,
                              CL_TYPE2* target_moments)
{

    //this is the slow O(p^4) method of translation
    CL_TYPE4 del = source_origin - target_origin;
    del.s3 = 0;

    CL_TYPE r = RadiusSingle(del);
    CL_TYPE costheta = CosThetaSingle(del);
    CL_TYPE phi = PhiSingle(del);

    //perform the summation weighted by the response functions
    CL_TYPE2 source;
    CL_TYPE2 response;
    CL_TYPE2 target;

    for(int j=0; j <= max_degree; j++)
    {
        for(int k=0; k<=j; k++)
        {
            int tsi = (j*(j+1))/2 + k;
            target.s0 = 0.0;
            target.s1 = 0.0;

            for(int n=0; n<=j; n++)
            {
                for(int m=-n; m<=n; m++)
                {
                    int j_minus_n = j-n;
                    int k_minus_m = k-m;

                    if( abs(k_minus_m) <= j_minus_n)
                    {
                        int ssi = j_minus_n*(j_minus_n+1)/2 + abs(k_minus_m);
                        source = source_moments[ssi];
                        if(k_minus_m > 0){source.s1 *= -1.0;};

                        CL_TYPE norm = a_coefficient[n*(n+1)/2 + abs(m)]*a_coefficient[ssi]/a_coefficient[tsi];
                        if( (k_minus_m)*m < 0)
                        {
                            norm *= pow( -1.0, min(abs(k_minus_m), abs(m)));
                        }

                        response.s0 = cos(-1*m*phi);
                        response.s1 = sin(-1*m*phi);
                        response *= norm;
                        response *= pow(r,n);
                        response *= ALP_nm(n,abs(m),costheta);

                        target.s0 += source.s0*response.s0 - source.s1*response.s1;
                        target.s1 += source.s0*response.s1 + source.s1*response.s0;
                    }
                }
            }

            target_moments[tsi] = target;

        }
    }


}


//______________________________________________________________________________

void
TranslateMomentsAlongZ( int max_degree,
                        __constant const CL_TYPE* a_coefficient,
                        __constant const CL_TYPE* axial_plm,
                        CL_TYPE r,
                        CL_TYPE2* source_moments,
                        CL_TYPE2* target_moments)
{

    //the axial translation is O(p^3)

    //size of the multipole moments
    int size = ((max_degree+1)*(max_degree+2))/2;

    //pre-multiply the source moments by their associated a_coefficient
    for(size_t i=0; i<size; i++)
    {
        source_moments[i] *= a_coefficient[i];
    }

    //perform the summation weighted by the response functions
    int target_si;
    int source_si;
    int scale_si;
    int j_minus_n;

    for(int j=0; j <= max_degree; j++)
    {
        for(int k=0; k <= j; k++)
        {
            target_si = j*(j+1)/2 + k;
            target_moments[target_si].s0 = 0.0;
            target_moments[target_si].s1 = 0.0;

            for(int n=0; n <= j; n++)
            {
                scale_si = n*(n+1)/2;
                j_minus_n = j-n;
                if(k <= j_minus_n)
                {
                    source_si = j_minus_n*(j_minus_n+1)/2 + k;
                    target_moments[target_si] += pown(r,n)*(axial_plm[scale_si])*(source_moments[source_si])*(a_coefficient[scale_si]);
                }
            }
        }
    }

    //post-divide the target moments by their associated a_coefficient
    for(size_t i=0; i<size; i++)
    {
        target_moments[i] *= (1.0/a_coefficient[i]);
    }
}


//______________________________________________________________________________

void
TranslateMultipoleMomentsFast( int max_degree,
                      __constant const CL_TYPE* a_coefficient,
                      __constant const CL_TYPE* axial_plm,
                      __constant const CL_TYPE* pinchon_j,
                      CL_TYPE4 source_origin,
                      CL_TYPE4 target_origin,
                      CL_TYPE* scratch1,
                      CL_TYPE* scratch2,
                      CL_TYPE2* source_moments,
                      CL_TYPE2* target_moments)

{
    //size of the multipole moments
    int size = ((max_degree+1)*(max_degree+2))/2;

    target_origin.s3 = 0.0;
    source_origin.s3 = 0.0;

    CL_TYPE4 del = source_origin - target_origin;// - source_origin;
    CL_TYPE r = length(del);
    CL_TYPE4 del_norm = normalize(del);
    CL_TYPE4 z_hat;
    z_hat.s0 = 0.0;
    z_hat.s1 = 0.0;
    z_hat.s2 = 1.0;
    z_hat.s3 = 0.0;

    //we want a rotation such that z' = del_norm
    //now lets compute the cross product of z_hat and del_norm (axis of the rotation)
    CL_TYPE4 rot_axis = cross(z_hat, del_norm);
    CL_TYPE sin_angle = length(rot_axis);
    CL_TYPE cos_angle = dot(z_hat, del_norm);

    if(sin_angle > 1e-3)
    {
        //compute the rotation matrix
        CL_TYPE4 normalized_rot_axis = normalize(rot_axis);
        CL_TYPE16 mx = MatrixFromAxisAngle(cos_angle, sin_angle, normalized_rot_axis);
        CL_TYPE R[9];

        R[rmsi(0,0)] = mx.s0;
        R[rmsi(0,1)] = mx.s1;
        R[rmsi(0,2)] = mx.s2;
        R[rmsi(1,0)] = mx.s3;
        R[rmsi(1,1)] = mx.s4;
        R[rmsi(1,2)] = mx.s5;
        R[rmsi(2,0)] = mx.s6;
        R[rmsi(2,1)] = mx.s7;
        R[rmsi(2,2)] = mx.s8;

        //compute the euler angles from the rotation matrix
        CL_TYPE4 euler_angles = EulerAnglesFromMatrixTranspose(R);
        CL_TYPE alpha = euler_angles.s0;
        CL_TYPE beta = euler_angles.s1;
        CL_TYPE gamma = euler_angles.s2;

        //now rotate the moments
        ApplyFullEulerRotation_ZYZ(max_degree, pinchon_j, alpha, beta, gamma, source_moments, scratch1, scratch2);

        //now translate the multipoles along z so they are about the origin we want,
        TranslateMomentsAlongZ(max_degree, a_coefficient, axial_plm, r, source_moments, target_moments);

        //now apply the inverse rotation
        euler_angles = EulerAnglesFromMatrix(R);
        alpha = euler_angles.s0;
        beta = euler_angles.s1;
        gamma = euler_angles.s2;

        //now rotate the moments
        ApplyFullEulerRotation_ZYZ(max_degree, pinchon_j, alpha, beta, gamma, target_moments, scratch1, scratch2);

        //take complex conjugate
        for(unsigned int i=0; i<size; i++)
        {
            target_moments[i].s1 *= -1.0;
        }

    }
    else
    {
        //compute the moment translation the slow way
        TranslateMultipoleMoments(max_degree, a_coefficient, source_origin, target_origin, source_moments, target_moments);
    }

}





#endif
