#ifndef KFMLineSegmentMultipole_Defined_H
#define KFMLineSegmentMultipole_Defined_H

#include "kEMField_defines.h"
#include "kEMField_KFMSphericalMultipoleMath.cl"

//returns the origin of the expansion
CL_TYPE4
LineSegmentMultipoleMoments(int max_degree,
                            CL_TYPE16 vertex,
                            CL_TYPE* scratch,
                            CL_TYPE2* moments)
{
    //read in the wire data
    CL_TYPE4 pA, pB;

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

    //compute the angular coordinates of pB w/ respect to pA
    CL_TYPE costheta, phi, radial_factor, dl, dm;
    costheta = CosTheta(pA, pB);
    phi = Phi(pA, pB);

    //compute assc. legendre poly. array in scratch space
    ALP_nm_array(max_degree, costheta, scratch);

    int si;
    CL_TYPE temp;
    for(int l=0; l <= max_degree; l++)
    {
        dl = l;
        radial_factor = pow(len, dl + 1.0)*(1.0/( dl + 1.0 ));

        for(int m=0; m <= l; m++)
        {
            dm = m;
            si = (l*(l+1))/2 + m;
            temp = inv_len*radial_factor*scratch[si];
            moments[si].s0 = temp*cos(dm*phi);
            moments[si].s1 = temp*sin(dm*phi);
        }
    }

    return pA;

}


#endif
