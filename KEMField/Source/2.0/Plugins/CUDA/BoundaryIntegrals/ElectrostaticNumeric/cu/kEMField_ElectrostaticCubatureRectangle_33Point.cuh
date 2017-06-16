#ifndef KEMFIELD_ELECTROSTATICCUBATURERECTANGLE_33POINT_CUH
#define KEMFIELD_ELECTROSTATICCUBATURERECTANGLE_33POINT_CUH

// CUDA kernel for rectangle boundary integrator with 33-point Gaussian cubature,
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
// * ER_Potential_Cub33P: 1024
// * ER_EField_Cub33P: 896
// * ER_EFieldAndPotential_Cub33P: 896

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

__constant__ CU_TYPE cuRectCub33w[33];

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE ER_Potential_Cub33P( const CU_TYPE* P,  const CU_TYPE* data )
{
    const CU_TYPE prefacStatic = data[0] * data[1] * M_ONEOVER_4PI_EPS0;
    CU_TYPE finalSum = 0.;

    const CU_TYPE a2 = 0.5*data[0];
    const CU_TYPE b2 = 0.5*data[1];

    const CU_TYPE cen0 = data[2] + (a2*data[5]) + (b2*data[8]);
    const CU_TYPE cen1 = data[3] + (a2*data[6]) + (b2*data[9]);
    const CU_TYPE cen2 = data[4] + (a2*data[7]) + (b2*data[10]);

    // 0

    CU_TYPE x = 0.00000000000000000000;
    CU_TYPE y = 0.00000000000000000000;

    CU_TYPE cub33Q[3] = {
            cen0 + (x*a2)*data[5] + (y*b2)*data[8],
            cen1 + (x*a2)*data[6] + (y*b2)*data[9],
            cen2 + (x*a2)*data[7] + (y*b2)*data[10]
    };

    finalSum += (cuRectCub33w[0]*OneOverR(cub33Q,P));

    // 1

    x = 0.77880971155441942252;
    y = 0.98348668243987226379;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[1]*OneOverR(cub33Q,P));

    // 2

    x = -0.77880971155441942252;
    y = -0.98348668243987226379;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[2]*OneOverR(cub33Q,P));

    // 3

    x = -0.98348668243987226379;
    y = 0.77880971155441942252;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[3]*OneOverR(cub33Q,P));

    // 4

    x = 0.98348668243987226379;
    y = -0.77880971155441942252;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[4]*OneOverR(cub33Q,P));

    // 5

    x = 0.95729769978630736566;
    y = 0.85955600564163892859;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[5]*OneOverR(cub33Q,P));

    // 6

    x = -0.95729769978630736566;
    y = -0.85955600564163892859;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[6]*OneOverR(cub33Q,P));

    // 7

    x = -0.85955600564163892859;
    y = 0.95729769978630736566;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[7]*OneOverR(cub33Q,P));

    // 8

    x = 0.85955600564163892859;
    y = -0.95729769978630736566;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[8]*OneOverR(cub33Q,P));

    // 9

    x = 0.13818345986246535375;
    y = 0.95892517028753485754;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[9]*OneOverR(cub33Q,P));

    // 10

    x = -0.13818345986246535375;
    y = -0.95892517028753485754;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[10]*OneOverR(cub33Q,P));

    // 11

    x = -0.95892517028753485754;
    y = 0.13818345986246535375;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[11]*OneOverR(cub33Q,P));

    // 12

    x = 0.95892517028753485754;
    y = -0.13818345986246535375;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[12]*OneOverR(cub33Q,P));

    // 13

    x = 0.94132722587292523695;
    y = 0.39073621612946100068;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[13]*OneOverR(cub33Q,P));

    // 14

    x = -0.94132722587292523695;
    y = -0.39073621612946100068;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[14]*OneOverR(cub33Q,P));

    // 15

    x = -0.39073621612946100068;
    y = 0.94132722587292523695;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[15]*OneOverR(cub33Q,P));

    // 16

    x = 0.39073621612946100068;
    y = -0.94132722587292523695;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[16]*OneOverR(cub33Q,P));

    // 17

    x = 0.47580862521827590507;
    y = 0.85007667369974857597;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[17]*OneOverR(cub33Q,P));

    // 18

    x = -0.47580862521827590507;
    y = -0.85007667369974857597;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[18]*OneOverR(cub33Q,P));

    // 19

    x = -0.85007667369974857597;
    y = 0.47580862521827590507;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[19]*OneOverR(cub33Q,P));

    // 20

    x = 0.85007667369974857597;
    y = -0.47580862521827590507;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[20]*OneOverR(cub33Q,P));

    // 21

    x = 0.75580535657208143627;
    y = 0.64782163718701073204;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[21]*OneOverR(cub33Q,P));

    // 22

    x = -0.75580535657208143627;
    y = -0.64782163718701073204;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[22]*OneOverR(cub33Q,P));

    // 23

    x = -0.64782163718701073204;
    y = 0.75580535657208143627;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[23]*OneOverR(cub33Q,P));

    // 24

    x = 0.64782163718701073204;
    y = -0.75580535657208143627;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[24]*OneOverR(cub33Q,P));

    // 25

    x = 0.69625007849174941396;
    y = 0.70741508996444936217e-1;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[25]*OneOverR(cub33Q,P));

    // 26

    x = -0.69625007849174941396;
    y = -0.70741508996444936217e-1;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[26]*OneOverR(cub33Q,P));

    // 27

    x = -0.70741508996444936217e-1;
    y = 0.69625007849174941396;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[27]*OneOverR(cub33Q,P));

    // 28

    x = 0.70741508996444936217e-1;
    y = -0.69625007849174941396;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[28]*OneOverR(cub33Q,P));

    // 29

    x = 0.34271655604040678941;
    y = 0.40930456169403884330;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[29]*OneOverR(cub33Q,P));

    // 30

    x = -0.34271655604040678941;
    y = -0.40930456169403884330;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[30]*OneOverR(cub33Q,P));

    // 31

    x = -0.40930456169403884330;
    y = 0.34271655604040678941;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[31]*OneOverR(cub33Q,P));

    // 32

    x = 0.40930456169403884330;
    y = -0.34271655604040678941;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub33w[32]*OneOverR(cub33Q,P));

    return (prefacStatic*finalSum);
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE4 ER_EField_Cub33P( const CU_TYPE* P,  const CU_TYPE* data )
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

    // 0

    CU_TYPE x = 0.00000000000000000000;
    CU_TYPE y = 0.00000000000000000000;

    CU_TYPE cub33Q[3] = {
            cen0 + (x*a2)*data[5] + (y*b2)*data[8],
            cen1 + (x*a2)*data[6] + (y*b2)*data[9],
            cen2 + (x*a2)*data[7] + (y*b2)*data[10]
    };

    CU_TYPE oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    CU_TYPE prefacDynamic = cuRectCub33w[0]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 1

    x = 0.77880971155441942252;
    y = 0.98348668243987226379;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[1]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 2

    x = -0.77880971155441942252;
    y = -0.98348668243987226379;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[2]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 3

    x = -0.98348668243987226379;
    y = 0.77880971155441942252;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[3]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 4

    x = 0.98348668243987226379;
    y = -0.77880971155441942252;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[4]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 5

    x = 0.95729769978630736566;
    y = 0.85955600564163892859;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[5]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 6

    x = -0.95729769978630736566;
    y = -0.85955600564163892859;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[6]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 7

    x = -0.85955600564163892859;
    y = 0.95729769978630736566;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[7]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 8

    x = 0.85955600564163892859;
    y = -0.95729769978630736566;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[8]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 9

    x = 0.13818345986246535375;
    y = 0.95892517028753485754;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[9]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 10

    x = -0.13818345986246535375;
    y = -0.95892517028753485754;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[10]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 11

    x = -0.95892517028753485754;
    y = 0.13818345986246535375;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[11]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 12

    x = 0.95892517028753485754;
    y = -0.13818345986246535375;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[12]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 13

    x = 0.94132722587292523695;
    y = 0.39073621612946100068;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[13]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 14

    x = -0.94132722587292523695;
    y = -0.39073621612946100068;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[14]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 15

    x = -0.39073621612946100068;
    y = 0.94132722587292523695;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[15]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 16

    x = 0.39073621612946100068;
    y = -0.94132722587292523695;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[16]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 17

    x = 0.47580862521827590507;
    y = 0.85007667369974857597;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[17]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 18

    x = -0.47580862521827590507;
    y = -0.85007667369974857597;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[18]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 19

    x = -0.85007667369974857597;
    y = 0.47580862521827590507;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[19]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 20

    x = 0.85007667369974857597;
    y = -0.47580862521827590507;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[20]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 21

    x = 0.75580535657208143627;
    y = 0.64782163718701073204;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[21]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 22

    x = -0.75580535657208143627;
    y = -0.64782163718701073204;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[22]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 23

    x = -0.64782163718701073204;
    y = 0.75580535657208143627;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[23]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 24

    x = 0.64782163718701073204;
    y = -0.75580535657208143627;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[24]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 25

    x = 0.69625007849174941396;
    y = 0.70741508996444936217e-1;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[25]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 26

    x = -0.69625007849174941396;
    y = -0.70741508996444936217e-1;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[26]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 27

    x = -0.70741508996444936217e-1;
    y = 0.69625007849174941396;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[27]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 28

    x = 0.70741508996444936217e-1;
    y = -0.69625007849174941396;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[28]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 29

    x = 0.34271655604040678941;
    y = 0.40930456169403884330;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[29]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 30

    x = -0.34271655604040678941;
    y = -0.40930456169403884330;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[30]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 31

    x = -0.40930456169403884330;
    y = 0.34271655604040678941;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[31]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 32

    x = 0.40930456169403884330;
    y = -0.34271655604040678941;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[32]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    return MAKECU4( prefacStatic*finalSum0, prefacStatic*finalSum1, prefacStatic*finalSum2, 0. );
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE4 ER_EFieldAndPotential_Cub33P( const CU_TYPE* P,  const CU_TYPE* data )
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

    // 0

    CU_TYPE x = 0.00000000000000000000;
    CU_TYPE y = 0.00000000000000000000;

    CU_TYPE cub33Q[3] = {
            cen0 + (x*a2)*data[5] + (y*b2)*data[8],
            cen1 + (x*a2)*data[6] + (y*b2)*data[9],
            cen2 + (x*a2)*data[7] + (y*b2)*data[10]
    };

    CU_TYPE oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    CU_TYPE prefacDynamic = cuRectCub33w[0]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[0]*OneOverR(cub33Q,P));

    // 1

    x = 0.77880971155441942252;
    y = 0.98348668243987226379;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[1]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[1]*OneOverR(cub33Q,P));

    // 2

    x = -0.77880971155441942252;
    y = -0.98348668243987226379;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[2]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[2]*OneOverR(cub33Q,P));

    // 3

    x = -0.98348668243987226379;
    y = 0.77880971155441942252;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[3]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[3]*OneOverR(cub33Q,P));

    // 4

    x = 0.98348668243987226379;
    y = -0.77880971155441942252;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[4]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[4]*OneOverR(cub33Q,P));

    // 5

    x = 0.95729769978630736566;
    y = 0.85955600564163892859;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[5]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[5]*OneOverR(cub33Q,P));

    // 6

    x = -0.95729769978630736566;
    y = -0.85955600564163892859;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[6]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[6]*OneOverR(cub33Q,P));

    // 7

    x = -0.85955600564163892859;
    y = 0.95729769978630736566;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[7]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[7]*OneOverR(cub33Q,P));

    // 8

    x = 0.85955600564163892859;
    y = -0.95729769978630736566;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[8]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[8]*OneOverR(cub33Q,P));

    // 9

    x = 0.13818345986246535375;
    y = 0.95892517028753485754;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[9]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[9]*OneOverR(cub33Q,P));

    // 10

    x = -0.13818345986246535375;
    y = -0.95892517028753485754;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[10]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[10]*OneOverR(cub33Q,P));

    // 11

    x = -0.95892517028753485754;
    y = 0.13818345986246535375;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[11]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[11]*OneOverR(cub33Q,P));

    // 12

    x = 0.95892517028753485754;
    y = -0.13818345986246535375;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[12]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[12]*OneOverR(cub33Q,P));

    // 13

    x = 0.94132722587292523695;
    y = 0.39073621612946100068;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[13]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[13]*OneOverR(cub33Q,P));

    // 14

    x = -0.94132722587292523695;
    y = -0.39073621612946100068;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[14]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[14]*OneOverR(cub33Q,P));

    // 15

    x = -0.39073621612946100068;
    y = 0.94132722587292523695;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[15]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[15]*OneOverR(cub33Q,P));

    // 16

    x = 0.39073621612946100068;
    y = -0.94132722587292523695;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[16]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[16]*OneOverR(cub33Q,P));

    // 17

    x = 0.47580862521827590507;
    y = 0.85007667369974857597;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[17]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[17]*OneOverR(cub33Q,P));

    // 18

    x = -0.47580862521827590507;
    y = -0.85007667369974857597;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[18]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[18]*OneOverR(cub33Q,P));

    // 19

    x = -0.85007667369974857597;
    y = 0.47580862521827590507;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[19]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[19]*OneOverR(cub33Q,P));

    // 20

    x = 0.85007667369974857597;
    y = -0.47580862521827590507;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[20]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[20]*OneOverR(cub33Q,P));

    // 21

    x = 0.75580535657208143627;
    y = 0.64782163718701073204;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[21]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[21]*OneOverR(cub33Q,P));

    // 22

    x = -0.75580535657208143627;
    y = -0.64782163718701073204;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[22]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[22]*OneOverR(cub33Q,P));

    // 23

    x = -0.64782163718701073204;
    y = 0.75580535657208143627;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[23]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[23]*OneOverR(cub33Q,P));

    // 24

    x = 0.64782163718701073204;
    y = -0.75580535657208143627;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[24]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[24]*OneOverR(cub33Q,P));

    // 25

    x = 0.69625007849174941396;
    y = 0.70741508996444936217e-1;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[25]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[25]*OneOverR(cub33Q,P));

    // 26

    x = -0.69625007849174941396;
    y = -0.70741508996444936217e-1;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[26]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[26]*OneOverR(cub33Q,P));

    // 27

    x = -0.70741508996444936217e-1;
    y = 0.69625007849174941396;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[27]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[27]*OneOverR(cub33Q,P));

    // 28

    x = 0.70741508996444936217e-1;
    y = -0.69625007849174941396;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[28]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[28]*OneOverR(cub33Q,P));

    // 29

    x = 0.34271655604040678941;
    y = 0.40930456169403884330;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[29]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[29]*OneOverR(cub33Q,P));

    // 30

    x = -0.34271655604040678941;
    y = -0.40930456169403884330;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[30]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[30]*OneOverR(cub33Q,P));

    // 31

    x = -0.40930456169403884330;
    y = 0.34271655604040678941;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[31]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[31]*OneOverR(cub33Q,P));

    // 32

    x = 0.40930456169403884330;
    y = -0.34271655604040678941;

    cub33Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub33Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub33Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuRectCub33w[32]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub33w[32]*OneOverR(cub33Q,P));

    return MAKECU4( prefacStatic*finalSum0, prefacStatic*finalSum1, prefacStatic*finalSum2, prefacStatic*finalSum3 );
}

#endif /* KEMFIELD_ELECTROSTATICCUBATURENUMERICRECTANGLE_33POINT_CUH */
