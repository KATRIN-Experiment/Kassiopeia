#ifndef KEMFIELD_ELECTROSTATICCUBATURERECTANGLE_7POINT_CUH
#define KEMFIELD_ELECTROSTATICCUBATURERECTANGLE_7POINT_CUH

// CUDA kernel for rectangle boundary integrator with 7-point Gaussian cubature,
// 5-1 7-point formula on p. 246 of Stroud book
// Detailed information on the cubature implementation can be found in the CPU code,
// class 'KElectrostaticCubatureRectangleIntegrator'.
// Author: Daniel Hilk
//
// This kernel version is optimized regarding thread block size and speed for compute
// devices providing only scalar units, but runs as well very efficient on devices
// providing vector units additionally. The optimal sizes for the used hardware
// will be set automatically.
//
// Recommended thread block sizes by CUDA Occupancy API for NVIDIA Tesla K40c:
// * ER_Potential_Cub7P: 1024
// * ER_EField_Cub7P: 768
// * ER_EFieldAndPotential_Cub7P: 768

#include "kEMField_Rectangle.cuh"
#include "kEMField_ElectrostaticCubature_CommonFunctions.cuh"

// Rectangle geometry definition (as defined by the streamers in KRectangle.hh):
//
// data[0]:     A
// data[1]:     B
// data[2..4]:  P0[0..2]
// data[5..7]:  N1[0..2]
// data[8..10]: N2[0..2]

//______________________________________________________________________________

// Gaussian weights

__constant__ CU_TYPE cuRectCub7w[7];

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE ER_Potential_Cub7P( const CU_TYPE* P,  const CU_TYPE* data )
{
    const CU_TYPE prefacStatic = data[0] * data[1] * M_ONEOVER_4PI_EPS0;
    CU_TYPE finalSum = 0.;

    const CU_TYPE a2 = 0.5*data[0];
    const CU_TYPE b2 = 0.5*data[1];

    const CU_TYPE cen0 = data[2] + (a2*data[5]) + (b2*data[8]);
    const CU_TYPE cen1 = data[3] + (a2*data[6]) + (b2*data[9]);
    const CU_TYPE cen2 = data[4] + (a2*data[7]) + (b2*data[10]);

    const CU_TYPE t1 = SQRT(14./15.);
    const CU_TYPE r1 = SQRT(3./5.);
    const CU_TYPE s1 = SQRT(1./3.);

    // 0

    CU_TYPE x1 = 0.;
    CU_TYPE y1 = 0.;

    CU_TYPE cub7Q[3] = {
            cen0 + (x1*a2)*data[5] + (y1*b2)*data[8],
            cen1 + (x1*a2)*data[6] + (y1*b2)*data[9],
            cen2 + (x1*a2)*data[7] + (y1*b2)*data[10]
    };

    finalSum += (cuRectCub7w[0]*OneOverR(cub7Q,P));

    // 1

    x1 = 0.;
    y1 = t1;

    cub7Q[0] = cen0 + (x1*a2)*data[5] + (y1*b2)*data[8];
    cub7Q[1] = cen1 + (x1*a2)*data[6] + (y1*b2)*data[9];
    cub7Q[2] = cen2 + (x1*a2)*data[7] + (y1*b2)*data[10];

    finalSum += (cuRectCub7w[1]*OneOverR(cub7Q,P));

    // 2

    x1 = 0.;
    y1 = -t1;

    cub7Q[0] = cen0 + (x1*a2)*data[5] + (y1*b2)*data[8];
    cub7Q[1] = cen1 + (x1*a2)*data[6] + (y1*b2)*data[9];
    cub7Q[2] = cen2 + (x1*a2)*data[7] + (y1*b2)*data[10];

    finalSum += (cuRectCub7w[2]*OneOverR(cub7Q,P));

    // 3

    x1 = r1;
    y1 = s1;

    cub7Q[0] = cen0 + (x1*a2)*data[5] + (y1*b2)*data[8];
    cub7Q[1] = cen1 + (x1*a2)*data[6] + (y1*b2)*data[9];
    cub7Q[2] = cen2 + (x1*a2)*data[7] + (y1*b2)*data[10];

    finalSum += (cuRectCub7w[3]*OneOverR(cub7Q,P));

    // 4

    x1 = r1;
    y1 = -s1;

    cub7Q[0] = cen0 + (x1*a2)*data[5] + (y1*b2)*data[8];
    cub7Q[1] = cen1 + (x1*a2)*data[6] + (y1*b2)*data[9];
    cub7Q[2] = cen2 + (x1*a2)*data[7] + (y1*b2)*data[10];

    finalSum += (cuRectCub7w[4]*OneOverR(cub7Q,P));

    // 5

    x1 = -r1;
    y1 = s1;

    cub7Q[0] = cen0 + (x1*a2)*data[5] + (y1*b2)*data[8];
    cub7Q[1] = cen1 + (x1*a2)*data[6] + (y1*b2)*data[9];
    cub7Q[2] = cen2 + (x1*a2)*data[7] + (y1*b2)*data[10];

    finalSum += (cuRectCub7w[5]*OneOverR(cub7Q,P));

    // 6

    x1 = -r1;
    y1 = -s1;

    cub7Q[0] = cen0 + (x1*a2)*data[5] + (y1*b2)*data[8];
    cub7Q[1] = cen1 + (x1*a2)*data[6] + (y1*b2)*data[9];
    cub7Q[2] = cen2 + (x1*a2)*data[7] + (y1*b2)*data[10];

    finalSum += (cuRectCub7w[6]*OneOverR(cub7Q,P));

    return (prefacStatic*finalSum);
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE4 ER_EField_Cub7P(const CU_TYPE* P,  const CU_TYPE* data)
{
    const CU_TYPE prefacStatic = data[0] * data[1] * M_ONEOVER_4PI_EPS0;
    CU_TYPE distVector[3] = {0., 0., 0.};
    CU_TYPE finalSum0 = 0.;
    CU_TYPE finalSum1 = 0.;
    CU_TYPE finalSum2 = 0.;

    const CU_TYPE a2 = 0.5*data[0];
    const CU_TYPE b2 = 0.5*data[1];

    const CU_TYPE cen0 = data[2] + (a2*data[5]) + (b2*data[8]);
    const CU_TYPE cen1 = data[3] + (a2*data[6]) + (b2*data[9]);
    const CU_TYPE cen2 = data[4] + (a2*data[7]) + (b2*data[10]);

    const CU_TYPE t1 = SQRT(14./15.);
    const CU_TYPE r1 = SQRT(3./5.);
    const CU_TYPE s1 = SQRT(1./3.);

    // 0

    CU_TYPE x1 = 0.;
    CU_TYPE y1 = 0.;

    CU_TYPE cub7Q[3] = {
            cen0 + (x1*a2)*data[5] + (y1*b2)*data[8],
            cen1 + (x1*a2)*data[6] + (y1*b2)*data[9],
            cen2 + (x1*a2)*data[7] + (y1*b2)*data[10]
    };

    CU_TYPE oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
    CU_TYPE prefacDynamic = cuRectCub7w[0]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 1

    x1 = 0.;
    y1 = t1;

    cub7Q[0] = cen0 + (x1*a2)*data[5] + (y1*b2)*data[8];
    cub7Q[1] = cen1 + (x1*a2)*data[6] + (y1*b2)*data[9];
    cub7Q[2] = cen2 + (x1*a2)*data[7] + (y1*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
    prefacDynamic = cuRectCub7w[1]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 2

    x1 = 0.;
    y1 = -t1;

    cub7Q[0] = cen0 + (x1*a2)*data[5] + (y1*b2)*data[8];
    cub7Q[1] = cen1 + (x1*a2)*data[6] + (y1*b2)*data[9];
    cub7Q[2] = cen2 + (x1*a2)*data[7] + (y1*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
    prefacDynamic = cuRectCub7w[2]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 3

    x1 = r1;
    y1 = s1;

    cub7Q[0] = cen0 + (x1*a2)*data[5] + (y1*b2)*data[8];
    cub7Q[1] = cen1 + (x1*a2)*data[6] + (y1*b2)*data[9];
    cub7Q[2] = cen2 + (x1*a2)*data[7] + (y1*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
    prefacDynamic = cuRectCub7w[3]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 4

    x1 = r1;
    y1 = -s1;

    cub7Q[0] = cen0 + (x1*a2)*data[5] + (y1*b2)*data[8];
    cub7Q[1] = cen1 + (x1*a2)*data[6] + (y1*b2)*data[9];
    cub7Q[2] = cen2 + (x1*a2)*data[7] + (y1*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
    prefacDynamic = cuRectCub7w[4]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 5

    x1 = -r1;
    y1 = s1;

    cub7Q[0] = cen0 + (x1*a2)*data[5] + (y1*b2)*data[8];
    cub7Q[1] = cen1 + (x1*a2)*data[6] + (y1*b2)*data[9];
    cub7Q[2] = cen2 + (x1*a2)*data[7] + (y1*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
    prefacDynamic = cuRectCub7w[5]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 6

    x1 = -r1;
    y1 = -s1;

    cub7Q[0] = cen0 + (x1*a2)*data[5] + (y1*b2)*data[8];
    cub7Q[1] = cen1 + (x1*a2)*data[6] + (y1*b2)*data[9];
    cub7Q[2] = cen2 + (x1*a2)*data[7] + (y1*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
    prefacDynamic = cuRectCub7w[6]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    return MAKECU4( prefacStatic*finalSum0, prefacStatic*finalSum1, prefacStatic*finalSum2, 0. );
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE4 ER_EFieldAndPotential_Cub7P(const CU_TYPE* P,  const CU_TYPE* data)
{
    const CU_TYPE prefacStatic = data[0] * data[1] * M_ONEOVER_4PI_EPS0;
    CU_TYPE distVector[3] = {0., 0., 0.};
    CU_TYPE finalSum0 = 0.;
    CU_TYPE finalSum1 = 0.;
    CU_TYPE finalSum2 = 0.;
    CU_TYPE finalSum3 = 0.;

    const CU_TYPE a2 = 0.5*data[0];
    const CU_TYPE b2 = 0.5*data[1];

    const CU_TYPE cen0 = data[2] + (a2*data[5]) + (b2*data[8]);
    const CU_TYPE cen1 = data[3] + (a2*data[6]) + (b2*data[9]);
    const CU_TYPE cen2 = data[4] + (a2*data[7]) + (b2*data[10]);

    const CU_TYPE t1 = SQRT(14./15.);
    const CU_TYPE r1 = SQRT(3./5.);
    const CU_TYPE s1 = SQRT(1./3.);

    // 0

    CU_TYPE x1 = 0.;
    CU_TYPE y1 = 0.;

    CU_TYPE cub7Q[3] = {
            cen0 + (x1*a2)*data[5] + (y1*b2)*data[8],
            cen1 + (x1*a2)*data[6] + (y1*b2)*data[9],
            cen2 + (x1*a2)*data[7] + (y1*b2)*data[10]
    };

    CU_TYPE oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
    CU_TYPE prefacDynamic = cuRectCub7w[0]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub7w[0]*OneOverR(cub7Q,P));

    // 1

    x1 = 0.;
    y1 = t1;

    cub7Q[0] = cen0 + (x1*a2)*data[5] + (y1*b2)*data[8];
    cub7Q[1] = cen1 + (x1*a2)*data[6] + (y1*b2)*data[9];
    cub7Q[2] = cen2 + (x1*a2)*data[7] + (y1*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
    prefacDynamic = cuRectCub7w[1]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub7w[1]*OneOverR(cub7Q,P));

    // 2

    x1 = 0.;
    y1 = -t1;

    cub7Q[0] = cen0 + (x1*a2)*data[5] + (y1*b2)*data[8];
    cub7Q[1] = cen1 + (x1*a2)*data[6] + (y1*b2)*data[9];
    cub7Q[2] = cen2 + (x1*a2)*data[7] + (y1*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
    prefacDynamic = cuRectCub7w[2]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub7w[2]*OneOverR(cub7Q,P));

    // 3

    x1 = r1;
    y1 = s1;

    cub7Q[0] = cen0 + (x1*a2)*data[5] + (y1*b2)*data[8];
    cub7Q[1] = cen1 + (x1*a2)*data[6] + (y1*b2)*data[9];
    cub7Q[2] = cen2 + (x1*a2)*data[7] + (y1*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
    prefacDynamic = cuRectCub7w[3]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub7w[3]*OneOverR(cub7Q,P));

    // 4

    x1 = r1;
    y1 = -s1;

    cub7Q[0] = cen0 + (x1*a2)*data[5] + (y1*b2)*data[8];
    cub7Q[1] = cen1 + (x1*a2)*data[6] + (y1*b2)*data[9];
    cub7Q[2] = cen2 + (x1*a2)*data[7] + (y1*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
    prefacDynamic = cuRectCub7w[4]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub7w[4]*OneOverR(cub7Q,P));

    // 5

    x1 = -r1;
    y1 = s1;

    cub7Q[0] = cen0 + (x1*a2)*data[5] + (y1*b2)*data[8];
    cub7Q[1] = cen1 + (x1*a2)*data[6] + (y1*b2)*data[9];
    cub7Q[2] = cen2 + (x1*a2)*data[7] + (y1*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
    prefacDynamic = cuRectCub7w[5]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub7w[5]*OneOverR(cub7Q,P));

    // 6

    x1 = -r1;
    y1 = -s1;

    cub7Q[0] = cen0 + (x1*a2)*data[5] + (y1*b2)*data[8];
    cub7Q[1] = cen1 + (x1*a2)*data[6] + (y1*b2)*data[9];
    cub7Q[2] = cen2 + (x1*a2)*data[7] + (y1*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
    prefacDynamic = cuRectCub7w[6]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub7w[6]*OneOverR(cub7Q,P));

    return MAKECU4( prefacStatic*finalSum0, prefacStatic*finalSum1, prefacStatic*finalSum2, prefacStatic*finalSum3 );
}

#endif /* KEMFIELD_ELECTROSTATICCUBATURERECTANGLE_7POINT_CUH */
