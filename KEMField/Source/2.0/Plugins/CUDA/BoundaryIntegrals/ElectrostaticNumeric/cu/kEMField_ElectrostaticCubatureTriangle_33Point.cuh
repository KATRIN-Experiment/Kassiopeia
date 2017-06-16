#ifndef KEMFIELD_ELECTROSTATICCUBATURETRIANGLE_33POINT_CUH
#define KEMFIELD_ELECTROSTATICCUBATURETRIANGLE_33POINT_CUH

// CUDA kernel for triangle boundary integrator with 33-point Gaussian cubature
// Detailed information on the cubature implementation can be found in the CPU code,
// class 'KElectrostaticCubatureTriangleIntegrator'.
// Author: Daniel Hilk
//
// This kernel version is optimized regarding thread block size and speed for compute
// devices providing only scalar units, but runs as well very efficient on devices
// providing vector units additionally. The optimal sizes for the used hardware
// will be set automatically.
//
// Recommended thread block sizes by CUDA Occupancy API for NVIDIA Tesla K40c:
// * ET_Potential_Cub33P: 384
// * ET_EField_Cub33P: 512
// * ET_EFieldAndPotential_Cub33P: 512

#include "kEMField_Triangle.cuh"
#include "kEMField_ElectrostaticCubature_CommonFunctions.cuh"

// Triangle geometry definition (as defined by the streamers in KTriangle.hh):
//
// data[0]:     A
// data[1]:     B
// data[2..4]:  P0[0..2]
// data[5..7]:  N1[0..2]
// data[8..10]: N2[0..2]

//______________________________________________________________________________

// barycentric (area) coordinates of the Gaussian points

__constant__ CU_TYPE cuTriCub33alpha[8];
__constant__ CU_TYPE cuTriCub33beta[8];
__constant__ CU_TYPE cuTriCub33gamma[8];

// Gaussian weights

__constant__ CU_TYPE cuTriCub33w[33];


//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE ET_Potential_Cub33P( const CU_TYPE* P,  const CU_TYPE* data )
{
    const CU_TYPE prefacStatic = Tri_Area( data ) * M_ONEOVER_4PI_EPS0;
    CU_TYPE finalSum = 0.;

    // triangle corner points

    const CU_TYPE A0 = data[2];
    const CU_TYPE A1 = data[3];
    const CU_TYPE A2 = data[4];

    const CU_TYPE B0 = data[2] + (data[0]*data[5]);
    const CU_TYPE B1 = data[3] + (data[0]*data[6]);
    const CU_TYPE B2 = data[4] + (data[0]*data[7]);

    const CU_TYPE C0 = data[2] + (data[1]*data[8]);
    const CU_TYPE C1 = data[3] + (data[1]*data[9]);
    const CU_TYPE C2 = data[4] + (data[1]*data[10]);

    // [0] alpha_0 A, beta_0 B, gamma_0 C

    CU_TYPE cub33Q[3] = {
            cuTriCub33alpha[0]*A0 + cuTriCub33beta[0]*B0 + cuTriCub33gamma[0]*C0,
            cuTriCub33alpha[0]*A1 + cuTriCub33beta[0]*B1 + cuTriCub33gamma[0]*C1,
            cuTriCub33alpha[0]*A2 + cuTriCub33beta[0]*B2 + cuTriCub33gamma[0]*C2 };

    finalSum += (cuTriCub33w[0]*OneOverR(cub33Q,P));

    // [1] beta_0 A, alpha_0 B, gamma_0 C

    cub33Q[0] = cuTriCub33beta[0]*A0 + cuTriCub33alpha[0]*B0 + cuTriCub33gamma[0]*C0;
    cub33Q[1] = cuTriCub33beta[0]*A1 + cuTriCub33alpha[0]*B1 + cuTriCub33gamma[0]*C1;
    cub33Q[2] = cuTriCub33beta[0]*A2 + cuTriCub33alpha[0]*B2 + cuTriCub33gamma[0]*C2;

    finalSum += (cuTriCub33w[1]*OneOverR(cub33Q,P));

    // [2] gamma_0 A, beta_0 B, alpha_0 C

    cub33Q[0] = cuTriCub33gamma[0]*A0 + cuTriCub33beta[0]*B0 + cuTriCub33alpha[0]*C0;
    cub33Q[1] = cuTriCub33gamma[0]*A1 + cuTriCub33beta[0]*B1 + cuTriCub33alpha[0]*C1;
    cub33Q[2] = cuTriCub33gamma[0]*A2 + cuTriCub33beta[0]*B2 + cuTriCub33alpha[0]*C2;

    finalSum += (cuTriCub33w[2]*OneOverR(cub33Q,P));

    // [3] alpha_1 A, beta_1 B, gamma_1 C

    cub33Q[0] = cuTriCub33alpha[1]*A0 + cuTriCub33beta[1]*B0 + cuTriCub33gamma[1]*C0;
    cub33Q[1] = cuTriCub33alpha[1]*A1 + cuTriCub33beta[1]*B1 + cuTriCub33gamma[1]*C1;
    cub33Q[2] = cuTriCub33alpha[1]*A2 + cuTriCub33beta[1]*B2 + cuTriCub33gamma[1]*C2;

    finalSum += (cuTriCub33w[3]*OneOverR(cub33Q,P));

    // [4] beta_1 A, alpha_1 B, gamma_1 C

    cub33Q[0] = cuTriCub33beta[1]*A0 + cuTriCub33alpha[1]*B0 + cuTriCub33gamma[1]*C0;
    cub33Q[1] = cuTriCub33beta[1]*A1 + cuTriCub33alpha[1]*B1 + cuTriCub33gamma[1]*C1;
    cub33Q[2] = cuTriCub33beta[1]*A2 + cuTriCub33alpha[1]*B2 + cuTriCub33gamma[1]*C2;

    finalSum += (cuTriCub33w[4]*OneOverR(cub33Q,P));

    // [5] gamma_1 A, beta_1 B, alpha_1 C

    cub33Q[0] = cuTriCub33gamma[1]*A0 + cuTriCub33beta[1]*B0 + cuTriCub33alpha[1]*C0;
    cub33Q[1] = cuTriCub33gamma[1]*A1 + cuTriCub33beta[1]*B1 + cuTriCub33alpha[1]*C1;
    cub33Q[2] = cuTriCub33gamma[1]*A2 + cuTriCub33beta[1]*B2 + cuTriCub33alpha[1]*C2;

    finalSum += (cuTriCub33w[5]*OneOverR(cub33Q,P));

    // [6] alpha_2 A, beta_2 B, gamma_2 C

    cub33Q[0] = cuTriCub33alpha[2]*A0 + cuTriCub33beta[2]*B0 + cuTriCub33gamma[2]*C0;
    cub33Q[1] = cuTriCub33alpha[2]*A1 + cuTriCub33beta[2]*B1 + cuTriCub33gamma[2]*C1;
    cub33Q[2] = cuTriCub33alpha[2]*A2 + cuTriCub33beta[2]*B2 + cuTriCub33gamma[2]*C2;

    finalSum += (cuTriCub33w[6]*OneOverR(cub33Q,P));

    // [7] beta_2 A, alpha_2 B, gamma_2 C

    cub33Q[0] = cuTriCub33beta[2]*A0 + cuTriCub33alpha[2]*B0 + cuTriCub33gamma[2]*C0;
    cub33Q[1] = cuTriCub33beta[2]*A1 + cuTriCub33alpha[2]*B1 + cuTriCub33gamma[2]*C1;
    cub33Q[2] = cuTriCub33beta[2]*A2 + cuTriCub33alpha[2]*B2 + cuTriCub33gamma[2]*C2;

    finalSum += (cuTriCub33w[7]*OneOverR(cub33Q,P));

    // [8] gamma_2 A, beta_2 B, alpha_2 C

    cub33Q[0] = cuTriCub33gamma[2]*A0 + cuTriCub33beta[2]*B0 + cuTriCub33alpha[2]*C0;
    cub33Q[1] = cuTriCub33gamma[2]*A1 + cuTriCub33beta[2]*B1 + cuTriCub33alpha[2]*C1;
    cub33Q[2] = cuTriCub33gamma[2]*A2 + cuTriCub33beta[2]*B2 + cuTriCub33alpha[2]*C2;

    finalSum += (cuTriCub33w[8]*OneOverR(cub33Q,P));

    // [9] alpha_3 A, beta_3 B, gamma_3 C

    cub33Q[0] = cuTriCub33alpha[3]*A0 + cuTriCub33beta[3]*B0 + cuTriCub33gamma[3]*C0;
    cub33Q[1] = cuTriCub33alpha[3]*A1 + cuTriCub33beta[3]*B1 + cuTriCub33gamma[3]*C1;
    cub33Q[2] = cuTriCub33alpha[3]*A2 + cuTriCub33beta[3]*B2 + cuTriCub33gamma[3]*C2;

    finalSum += (cuTriCub33w[9]*OneOverR(cub33Q,P));

    // [10] beta_3 A, alpha_3 B, gamma_3 C

    cub33Q[0] = cuTriCub33beta[3]*A0 + cuTriCub33alpha[3]*B0 + cuTriCub33gamma[3]*C0;
    cub33Q[1] = cuTriCub33beta[3]*A1 + cuTriCub33alpha[3]*B1 + cuTriCub33gamma[3]*C1;
    cub33Q[2] = cuTriCub33beta[3]*A2 + cuTriCub33alpha[3]*B2 + cuTriCub33gamma[3]*C2;

    finalSum += (cuTriCub33w[10]*OneOverR(cub33Q,P));

    // [11] gamma_3 A, beta_3 B, alpha_3 C

    cub33Q[0] = cuTriCub33gamma[3]*A0 + cuTriCub33beta[3]*B0 + cuTriCub33alpha[3]*C0;
    cub33Q[1] = cuTriCub33gamma[3]*A1 + cuTriCub33beta[3]*B1 + cuTriCub33alpha[3]*C1;
    cub33Q[2] = cuTriCub33gamma[3]*A2 + cuTriCub33beta[3]*B2 + cuTriCub33alpha[3]*C2;

    finalSum += (cuTriCub33w[11]*OneOverR(cub33Q,P));

    // [12] alpha_4 A, beta_4 B, gamma_4 C

    cub33Q[0] = cuTriCub33alpha[4]*A0 + cuTriCub33beta[4]*B0 + cuTriCub33gamma[4]*C0;
    cub33Q[1] = cuTriCub33alpha[4]*A1 + cuTriCub33beta[4]*B1 + cuTriCub33gamma[4]*C1;
    cub33Q[2] = cuTriCub33alpha[4]*A2 + cuTriCub33beta[4]*B2 + cuTriCub33gamma[4]*C2;

    finalSum += (cuTriCub33w[12]*OneOverR(cub33Q,P));

    // [13] beta_4 A, alpha_4 B, gamma_4 C

    cub33Q[0] = cuTriCub33beta[4]*A0 + cuTriCub33alpha[4]*B0 + cuTriCub33gamma[4]*C0;
    cub33Q[1] = cuTriCub33beta[4]*A1 + cuTriCub33alpha[4]*B1 + cuTriCub33gamma[4]*C1;
    cub33Q[2] = cuTriCub33beta[4]*A2 + cuTriCub33alpha[4]*B2 + cuTriCub33gamma[4]*C2;

    finalSum += (cuTriCub33w[13]*OneOverR(cub33Q,P));

    // [14] gamma_4 A, beta_4 B, alpha_4 C

    cub33Q[0] = cuTriCub33gamma[4]*A0 + cuTriCub33beta[4]*B0 + cuTriCub33alpha[4]*C0;
    cub33Q[1] = cuTriCub33gamma[4]*A1 + cuTriCub33beta[4]*B1 + cuTriCub33alpha[4]*C1;
    cub33Q[2] = cuTriCub33gamma[4]*A2 + cuTriCub33beta[4]*B2 + cuTriCub33alpha[4]*C2;

    finalSum += (cuTriCub33w[14]*OneOverR(cub33Q,P));

    // [15] alpha_5 A, beta_5 B, gamma_5 C

    cub33Q[0] = cuTriCub33alpha[5]*A0 + cuTriCub33beta[5]*B0 + cuTriCub33gamma[5]*C0;
    cub33Q[1] = cuTriCub33alpha[5]*A1 + cuTriCub33beta[5]*B1 + cuTriCub33gamma[5]*C1;
    cub33Q[2] = cuTriCub33alpha[5]*A2 + cuTriCub33beta[5]*B2 + cuTriCub33gamma[5]*C2;

    finalSum += (cuTriCub33w[15]*OneOverR(cub33Q,P));

    // [16] beta_5 A, alpha_5 B, gamma_5 C

    cub33Q[0] = cuTriCub33beta[5]*A0 + cuTriCub33alpha[5]*B0 + cuTriCub33gamma[5]*C0;
    cub33Q[1] = cuTriCub33beta[5]*A1 + cuTriCub33alpha[5]*B1 + cuTriCub33gamma[5]*C1;
    cub33Q[2] = cuTriCub33beta[5]*A2 + cuTriCub33alpha[5]*B2 + cuTriCub33gamma[5]*C2;

    finalSum += (cuTriCub33w[16]*OneOverR(cub33Q,P));

    // [17] gamma_5 A, beta_5 B, alpha_5 C

    cub33Q[0] = cuTriCub33gamma[5]*A0 + cuTriCub33beta[5]*B0 + cuTriCub33alpha[5]*C0;
    cub33Q[1] = cuTriCub33gamma[5]*A1 + cuTriCub33beta[5]*B1 + cuTriCub33alpha[5]*C1;
    cub33Q[2] = cuTriCub33gamma[5]*A2 + cuTriCub33beta[5]*B2 + cuTriCub33alpha[5]*C2;

    finalSum += (cuTriCub33w[17]*OneOverR(cub33Q,P));

    // [18] alpha_5 A, gamma_5 B, beta_5 C

    cub33Q[0] = cuTriCub33alpha[5]*A0 + cuTriCub33gamma[5]*B0 + cuTriCub33beta[5]*C0;
    cub33Q[1] = cuTriCub33alpha[5]*A1 + cuTriCub33gamma[5]*B1 + cuTriCub33beta[5]*C1;
    cub33Q[2] = cuTriCub33alpha[5]*A2 + cuTriCub33gamma[5]*B2 + cuTriCub33beta[5]*C2;

    finalSum += (cuTriCub33w[18]*OneOverR(cub33Q,P));

    // [19] gamma_5 A, alpha_5 B, beta_5 C

    cub33Q[0] = cuTriCub33gamma[5]*A0 + cuTriCub33alpha[5]*B0 + cuTriCub33beta[5]*C0;
    cub33Q[1] = cuTriCub33gamma[5]*A1 + cuTriCub33alpha[5]*B1 + cuTriCub33beta[5]*C1;
    cub33Q[2] = cuTriCub33gamma[5]*A2 + cuTriCub33alpha[5]*B2 + cuTriCub33beta[5]*C2;

    finalSum += (cuTriCub33w[19]*OneOverR(cub33Q,P));

    // [20] beta_5 A, gamma_5 B, alpha_5 C

    cub33Q[0] = cuTriCub33beta[5]*A0 + cuTriCub33gamma[5]*B0 + cuTriCub33alpha[5]*C0;
    cub33Q[1] = cuTriCub33beta[5]*A1 + cuTriCub33gamma[5]*B1 + cuTriCub33alpha[5]*C1;
    cub33Q[2] = cuTriCub33beta[5]*A2 + cuTriCub33gamma[5]*B2 + cuTriCub33alpha[5]*C2;

    finalSum += (cuTriCub33w[20]*OneOverR(cub33Q,P));

    // [21] alpha_6 A, beta_6 B, gamma_6 C

    cub33Q[0] = cuTriCub33alpha[6]*A0 + cuTriCub33beta[6]*B0 + cuTriCub33gamma[6]*C0;
    cub33Q[1] = cuTriCub33alpha[6]*A1 + cuTriCub33beta[6]*B1 + cuTriCub33gamma[6]*C1;
    cub33Q[2] = cuTriCub33alpha[6]*A2 + cuTriCub33beta[6]*B2 + cuTriCub33gamma[6]*C2;

    finalSum += (cuTriCub33w[21]*OneOverR(cub33Q,P));

    // [22] beta_6 A, alpha_6 B, gamma_6 C

    cub33Q[0] = cuTriCub33beta[6]*A0 + cuTriCub33alpha[6]*B0 + cuTriCub33gamma[6]*C0;
    cub33Q[1] = cuTriCub33beta[6]*A1 + cuTriCub33alpha[6]*B1 + cuTriCub33gamma[6]*C1;
    cub33Q[2] = cuTriCub33beta[6]*A2 + cuTriCub33alpha[6]*B2 + cuTriCub33gamma[6]*C2;

    finalSum += (cuTriCub33w[22]*OneOverR(cub33Q,P));

    // [23] gamma_6 A, beta_6 B, alpha_6 C

    cub33Q[0] = cuTriCub33gamma[6]*A0 + cuTriCub33beta[6]*B0 + cuTriCub33alpha[6]*C0;
    cub33Q[1] = cuTriCub33gamma[6]*A1 + cuTriCub33beta[6]*B1 + cuTriCub33alpha[6]*C1;
    cub33Q[2] = cuTriCub33gamma[6]*A2 + cuTriCub33beta[6]*B2 + cuTriCub33alpha[6]*C2;

    finalSum += (cuTriCub33w[23]*OneOverR(cub33Q,P));

    // [24] alpha_6 A, gamma_6 B, beta_6 C

    cub33Q[0] = cuTriCub33alpha[6]*A0 + cuTriCub33gamma[6]*B0 + cuTriCub33beta[6]*C0;
    cub33Q[1] = cuTriCub33alpha[6]*A1 + cuTriCub33gamma[6]*B1 + cuTriCub33beta[6]*C1;
    cub33Q[2] = cuTriCub33alpha[6]*A2 + cuTriCub33gamma[6]*B2 + cuTriCub33beta[6]*C2;

    finalSum += (cuTriCub33w[24]*OneOverR(cub33Q,P));

    // [25] gamma_6 A, alpha_6 B, beta_6 C

    cub33Q[0] = cuTriCub33gamma[6]*A0 + cuTriCub33alpha[6]*B0 + cuTriCub33beta[6]*C0;
    cub33Q[1] = cuTriCub33gamma[6]*A1 + cuTriCub33alpha[6]*B1 + cuTriCub33beta[6]*C1;
    cub33Q[2] = cuTriCub33gamma[6]*A2 + cuTriCub33alpha[6]*B2 + cuTriCub33beta[6]*C2;

    finalSum += (cuTriCub33w[25]*OneOverR(cub33Q,P));

    // [26] beta_6 A, gamma_6 B, alpha_6 C

    cub33Q[0] = cuTriCub33beta[6]*A0 + cuTriCub33gamma[6]*B0 + cuTriCub33alpha[6]*C0;
    cub33Q[1] = cuTriCub33beta[6]*A1 + cuTriCub33gamma[6]*B1 + cuTriCub33alpha[6]*C1;
    cub33Q[2] = cuTriCub33beta[6]*A2 + cuTriCub33gamma[6]*B2 + cuTriCub33alpha[6]*C2;

    finalSum += (cuTriCub33w[26]*OneOverR(cub33Q,P));

    // [27] alpha_7 A, beta_7 B, gamma_7 C

    cub33Q[0] = cuTriCub33alpha[7]*A0 + cuTriCub33beta[7]*B0 + cuTriCub33gamma[7]*C0;
    cub33Q[1] = cuTriCub33alpha[7]*A1 + cuTriCub33beta[7]*B1 + cuTriCub33gamma[7]*C1;
    cub33Q[2] = cuTriCub33alpha[7]*A2 + cuTriCub33beta[7]*B2 + cuTriCub33gamma[7]*C2;

    finalSum += (cuTriCub33w[27]*OneOverR(cub33Q,P));

    // [28] beta_7 A, alpha_7 B, gamma_7 C

    cub33Q[0] = cuTriCub33beta[7]*A0 + cuTriCub33alpha[7]*B0 + cuTriCub33gamma[7]*C0;
    cub33Q[1] = cuTriCub33beta[7]*A1 + cuTriCub33alpha[7]*B1 + cuTriCub33gamma[7]*C1;
    cub33Q[2] = cuTriCub33beta[7]*A2 + cuTriCub33alpha[7]*B2 + cuTriCub33gamma[7]*C2;

    finalSum += (cuTriCub33w[28]*OneOverR(cub33Q,P));

    // [29] gamma_7 A, beta_7 B, alpha_7 C

    cub33Q[0] = cuTriCub33gamma[7]*A0 + cuTriCub33beta[7]*B0 + cuTriCub33alpha[7]*C0;
    cub33Q[1] = cuTriCub33gamma[7]*A1 + cuTriCub33beta[7]*B1 + cuTriCub33alpha[7]*C1;
    cub33Q[2] = cuTriCub33gamma[7]*A2 + cuTriCub33beta[7]*B2 + cuTriCub33alpha[7]*C2;

    finalSum += (cuTriCub33w[29]*OneOverR(cub33Q,P));

    // [30] alpha_7 A, gamma_7 B, beta_7 C

    cub33Q[0] = cuTriCub33alpha[7]*A0 + cuTriCub33gamma[7]*B0 + cuTriCub33beta[7]*C0;
    cub33Q[1] = cuTriCub33alpha[7]*A1 + cuTriCub33gamma[7]*B1 + cuTriCub33beta[7]*C1;
    cub33Q[2] = cuTriCub33alpha[7]*A2 + cuTriCub33gamma[7]*B2 + cuTriCub33beta[7]*C2;

    finalSum += (cuTriCub33w[30]*OneOverR(cub33Q,P));

    // [31] gamma_7 A, alpha_7 B, beta_7 C

    cub33Q[0] = cuTriCub33gamma[7]*A0 + cuTriCub33alpha[7]*B0 + cuTriCub33beta[7]*C0;
    cub33Q[1] = cuTriCub33gamma[7]*A1 + cuTriCub33alpha[7]*B1 + cuTriCub33beta[7]*C1;
    cub33Q[2] = cuTriCub33gamma[7]*A2 + cuTriCub33alpha[7]*B2 + cuTriCub33beta[7]*C2;

    finalSum += (cuTriCub33w[31]*OneOverR(cub33Q,P));

    // [32] beta_7 A, gamma_7 B, alpha_7 C

    cub33Q[0] = cuTriCub33beta[7]*A0 + cuTriCub33gamma[7]*B0 + cuTriCub33alpha[7]*C0;
    cub33Q[1] = cuTriCub33beta[7]*A1 + cuTriCub33gamma[7]*B1 + cuTriCub33alpha[7]*C1;
    cub33Q[2] = cuTriCub33beta[7]*A2 + cuTriCub33gamma[7]*B2 + cuTriCub33alpha[7]*C2;

    finalSum += (cuTriCub33w[32]*OneOverR(cub33Q,P));

    return (prefacStatic*finalSum);
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE4 ET_EField_Cub33P( const CU_TYPE* P,  const CU_TYPE* data )
{
    const CU_TYPE prefacStatic = Tri_Area( data ) * M_ONEOVER_4PI_EPS0;
    CU_TYPE distVector[3] = {0., 0., 0.};
    CU_TYPE finalSum0 = 0.;
    CU_TYPE finalSum1 = 0.;
    CU_TYPE finalSum2 = 0.;

    // triangle corner points

    const CU_TYPE A0 = data[2];
    const CU_TYPE A1 = data[3];
    const CU_TYPE A2 = data[4];

    const CU_TYPE B0 = data[2] + (data[0]*data[5]);
    const CU_TYPE B1 = data[3] + (data[0]*data[6]);
    const CU_TYPE B2 = data[4] + (data[0]*data[7]);

    const CU_TYPE C0 = data[2] + (data[1]*data[8]);
    const CU_TYPE C1 = data[3] + (data[1]*data[9]);
    const CU_TYPE C2 = data[4] + (data[1]*data[10]);

    // [0] alpha_0 A, beta_0 B, gamma_0 C

    CU_TYPE cub33Q[3] = {
            cuTriCub33alpha[0]*A0 + cuTriCub33beta[0]*B0 + cuTriCub33gamma[0]*C0,
            cuTriCub33alpha[0]*A1 + cuTriCub33beta[0]*B1 + cuTriCub33gamma[0]*C1,
            cuTriCub33alpha[0]*A2 + cuTriCub33beta[0]*B2 + cuTriCub33gamma[0]*C2 };

    CU_TYPE oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    CU_TYPE prefacDynamic = cuTriCub33w[0]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [1] beta_0 A, alpha_0 B, gamma_0 C

    cub33Q[0] = cuTriCub33beta[0]*A0 + cuTriCub33alpha[0]*B0 + cuTriCub33gamma[0]*C0;
    cub33Q[1] = cuTriCub33beta[0]*A1 + cuTriCub33alpha[0]*B1 + cuTriCub33gamma[0]*C1;
    cub33Q[2] = cuTriCub33beta[0]*A2 + cuTriCub33alpha[0]*B2 + cuTriCub33gamma[0]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[1]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [2] gamma_0 A, beta_0 B, alpha_0 C

    cub33Q[0] = cuTriCub33gamma[0]*A0 + cuTriCub33beta[0]*B0 + cuTriCub33alpha[0]*C0;
    cub33Q[1] = cuTriCub33gamma[0]*A1 + cuTriCub33beta[0]*B1 + cuTriCub33alpha[0]*C1;
    cub33Q[2] = cuTriCub33gamma[0]*A2 + cuTriCub33beta[0]*B2 + cuTriCub33alpha[0]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[2]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [3] alpha_1 A, beta_1 B, gamma_1 C

    cub33Q[0] = cuTriCub33alpha[1]*A0 + cuTriCub33beta[1]*B0 + cuTriCub33gamma[1]*C0;
    cub33Q[1] = cuTriCub33alpha[1]*A1 + cuTriCub33beta[1]*B1 + cuTriCub33gamma[1]*C1;
    cub33Q[2] = cuTriCub33alpha[1]*A2 + cuTriCub33beta[1]*B2 + cuTriCub33gamma[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[3]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [4] beta_1 A, alpha_1 B, gamma_1 C

    cub33Q[0] = cuTriCub33beta[1]*A0 + cuTriCub33alpha[1]*B0 + cuTriCub33gamma[1]*C0;
    cub33Q[1] = cuTriCub33beta[1]*A1 + cuTriCub33alpha[1]*B1 + cuTriCub33gamma[1]*C1;
    cub33Q[2] = cuTriCub33beta[1]*A2 + cuTriCub33alpha[1]*B2 + cuTriCub33gamma[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[4]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [5] gamma_1 A, beta_1 B, alpha_1 C

    cub33Q[0] = cuTriCub33gamma[1]*A0 + cuTriCub33beta[1]*B0 + cuTriCub33alpha[1]*C0;
    cub33Q[1] = cuTriCub33gamma[1]*A1 + cuTriCub33beta[1]*B1 + cuTriCub33alpha[1]*C1;
    cub33Q[2] = cuTriCub33gamma[1]*A2 + cuTriCub33beta[1]*B2 + cuTriCub33alpha[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[5]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [6] alpha_2 A, beta_2 B, gamma_2 C

    cub33Q[0] = cuTriCub33alpha[2]*A0 + cuTriCub33beta[2]*B0 + cuTriCub33gamma[2]*C0;
    cub33Q[1] = cuTriCub33alpha[2]*A1 + cuTriCub33beta[2]*B1 + cuTriCub33gamma[2]*C1;
    cub33Q[2] = cuTriCub33alpha[2]*A2 + cuTriCub33beta[2]*B2 + cuTriCub33gamma[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[6]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [7] beta_2 A, alpha_2 B, gamma_2 C

    cub33Q[0] = cuTriCub33beta[2]*A0 + cuTriCub33alpha[2]*B0 + cuTriCub33gamma[2]*C0;
    cub33Q[1] = cuTriCub33beta[2]*A1 + cuTriCub33alpha[2]*B1 + cuTriCub33gamma[2]*C1;
    cub33Q[2] = cuTriCub33beta[2]*A2 + cuTriCub33alpha[2]*B2 + cuTriCub33gamma[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[7]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [8] gamma_2 A, beta_2 B, alpha_2 C

    cub33Q[0] = cuTriCub33gamma[2]*A0 + cuTriCub33beta[2]*B0 + cuTriCub33alpha[2]*C0;
    cub33Q[1] = cuTriCub33gamma[2]*A1 + cuTriCub33beta[2]*B1 + cuTriCub33alpha[2]*C1;
    cub33Q[2] = cuTriCub33gamma[2]*A2 + cuTriCub33beta[2]*B2 + cuTriCub33alpha[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[8]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [9] alpha_3 A, beta_3 B, gamma_3 C

    cub33Q[0] = cuTriCub33alpha[3]*A0 + cuTriCub33beta[3]*B0 + cuTriCub33gamma[3]*C0;
    cub33Q[1] = cuTriCub33alpha[3]*A1 + cuTriCub33beta[3]*B1 + cuTriCub33gamma[3]*C1;
    cub33Q[2] = cuTriCub33alpha[3]*A2 + cuTriCub33beta[3]*B2 + cuTriCub33gamma[3]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[9]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [10] beta_3 A, alpha_3 B, gamma_3 C

    cub33Q[0] = cuTriCub33beta[3]*A0 + cuTriCub33alpha[3]*B0 + cuTriCub33gamma[3]*C0;
    cub33Q[1] = cuTriCub33beta[3]*A1 + cuTriCub33alpha[3]*B1 + cuTriCub33gamma[3]*C1;
    cub33Q[2] = cuTriCub33beta[3]*A2 + cuTriCub33alpha[3]*B2 + cuTriCub33gamma[3]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[10]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [11] gamma_3 A, beta_3 B, alpha_3 C

    cub33Q[0] = cuTriCub33gamma[3]*A0 + cuTriCub33beta[3]*B0 + cuTriCub33alpha[3]*C0;
    cub33Q[1] = cuTriCub33gamma[3]*A1 + cuTriCub33beta[3]*B1 + cuTriCub33alpha[3]*C1;
    cub33Q[2] = cuTriCub33gamma[3]*A2 + cuTriCub33beta[3]*B2 + cuTriCub33alpha[3]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[11]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [12] alpha_4 A, beta_4 B, gamma_4 C

    cub33Q[0] = cuTriCub33alpha[4]*A0 + cuTriCub33beta[4]*B0 + cuTriCub33gamma[4]*C0;
    cub33Q[1] = cuTriCub33alpha[4]*A1 + cuTriCub33beta[4]*B1 + cuTriCub33gamma[4]*C1;
    cub33Q[2] = cuTriCub33alpha[4]*A2 + cuTriCub33beta[4]*B2 + cuTriCub33gamma[4]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[12]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [13] beta_4 A, alpha_4 B, gamma_4 C

    cub33Q[0] = cuTriCub33beta[4]*A0 + cuTriCub33alpha[4]*B0 + cuTriCub33gamma[4]*C0;
    cub33Q[1] = cuTriCub33beta[4]*A1 + cuTriCub33alpha[4]*B1 + cuTriCub33gamma[4]*C1;
    cub33Q[2] = cuTriCub33beta[4]*A2 + cuTriCub33alpha[4]*B2 + cuTriCub33gamma[4]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[13]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [14] gamma_4 A, beta_4 B, alpha_4 C

    cub33Q[0] = cuTriCub33gamma[4]*A0 + cuTriCub33beta[4]*B0 + cuTriCub33alpha[4]*C0;
    cub33Q[1] = cuTriCub33gamma[4]*A1 + cuTriCub33beta[4]*B1 + cuTriCub33alpha[4]*C1;
    cub33Q[2] = cuTriCub33gamma[4]*A2 + cuTriCub33beta[4]*B2 + cuTriCub33alpha[4]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[14]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [15] alpha_5 A, beta_5 B, gamma_5 C

    cub33Q[0] = cuTriCub33alpha[5]*A0 + cuTriCub33beta[5]*B0 + cuTriCub33gamma[5]*C0;
    cub33Q[1] = cuTriCub33alpha[5]*A1 + cuTriCub33beta[5]*B1 + cuTriCub33gamma[5]*C1;
    cub33Q[2] = cuTriCub33alpha[5]*A2 + cuTriCub33beta[5]*B2 + cuTriCub33gamma[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[15]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [16] beta_5 A, alpha_5 B, gamma_5 C

    cub33Q[0] = cuTriCub33beta[5]*A0 + cuTriCub33alpha[5]*B0 + cuTriCub33gamma[5]*C0;
    cub33Q[1] = cuTriCub33beta[5]*A1 + cuTriCub33alpha[5]*B1 + cuTriCub33gamma[5]*C1;
    cub33Q[2] = cuTriCub33beta[5]*A2 + cuTriCub33alpha[5]*B2 + cuTriCub33gamma[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[16]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [17] gamma_5 A, beta_5 B, alpha_5 C

    cub33Q[0] = cuTriCub33gamma[5]*A0 + cuTriCub33beta[5]*B0 + cuTriCub33alpha[5]*C0;
    cub33Q[1] = cuTriCub33gamma[5]*A1 + cuTriCub33beta[5]*B1 + cuTriCub33alpha[5]*C1;
    cub33Q[2] = cuTriCub33gamma[5]*A2 + cuTriCub33beta[5]*B2 + cuTriCub33alpha[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[17]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [18] alpha_5 A, gamma_5 B, beta_5 C

    cub33Q[0] = cuTriCub33alpha[5]*A0 + cuTriCub33gamma[5]*B0 + cuTriCub33beta[5]*C0;
    cub33Q[1] = cuTriCub33alpha[5]*A1 + cuTriCub33gamma[5]*B1 + cuTriCub33beta[5]*C1;
    cub33Q[2] = cuTriCub33alpha[5]*A2 + cuTriCub33gamma[5]*B2 + cuTriCub33beta[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[18]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [19] gamma_5 A, alpha_5 B, beta_5 C

    cub33Q[0] = cuTriCub33gamma[5]*A0 + cuTriCub33alpha[5]*B0 + cuTriCub33beta[5]*C0;
    cub33Q[1] = cuTriCub33gamma[5]*A1 + cuTriCub33alpha[5]*B1 + cuTriCub33beta[5]*C1;
    cub33Q[2] = cuTriCub33gamma[5]*A2 + cuTriCub33alpha[5]*B2 + cuTriCub33beta[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[19]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [20] beta_5 A, gamma_5 B, alpha_5 C

    cub33Q[0] = cuTriCub33beta[5]*A0 + cuTriCub33gamma[5]*B0 + cuTriCub33alpha[5]*C0;
    cub33Q[1] = cuTriCub33beta[5]*A1 + cuTriCub33gamma[5]*B1 + cuTriCub33alpha[5]*C1;
    cub33Q[2] = cuTriCub33beta[5]*A2 + cuTriCub33gamma[5]*B2 + cuTriCub33alpha[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[20]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [21] alpha_6 A, beta_6 B, gamma_6 C

    cub33Q[0] = cuTriCub33alpha[6]*A0 + cuTriCub33beta[6]*B0 + cuTriCub33gamma[6]*C0;
    cub33Q[1] = cuTriCub33alpha[6]*A1 + cuTriCub33beta[6]*B1 + cuTriCub33gamma[6]*C1;
    cub33Q[2] = cuTriCub33alpha[6]*A2 + cuTriCub33beta[6]*B2 + cuTriCub33gamma[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[21]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [22] beta_6 A, alpha_6 B, gamma_6 C

    cub33Q[0] = cuTriCub33beta[6]*A0 + cuTriCub33alpha[6]*B0 + cuTriCub33gamma[6]*C0;
    cub33Q[1] = cuTriCub33beta[6]*A1 + cuTriCub33alpha[6]*B1 + cuTriCub33gamma[6]*C1;
    cub33Q[2] = cuTriCub33beta[6]*A2 + cuTriCub33alpha[6]*B2 + cuTriCub33gamma[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[22]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [23] gamma_6 A, beta_6 B, alpha_6 C

    cub33Q[0] = cuTriCub33gamma[6]*A0 + cuTriCub33beta[6]*B0 + cuTriCub33alpha[6]*C0;
    cub33Q[1] = cuTriCub33gamma[6]*A1 + cuTriCub33beta[6]*B1 + cuTriCub33alpha[6]*C1;
    cub33Q[2] = cuTriCub33gamma[6]*A2 + cuTriCub33beta[6]*B2 + cuTriCub33alpha[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[23]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [24] alpha_6 A, gamma_6 B, beta_6 C

    cub33Q[0] = cuTriCub33alpha[6]*A0 + cuTriCub33gamma[6]*B0 + cuTriCub33beta[6]*C0;
    cub33Q[1] = cuTriCub33alpha[6]*A1 + cuTriCub33gamma[6]*B1 + cuTriCub33beta[6]*C1;
    cub33Q[2] = cuTriCub33alpha[6]*A2 + cuTriCub33gamma[6]*B2 + cuTriCub33beta[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[24]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [25] gamma_6 A, alpha_6 B, beta_6 C

    cub33Q[0] = cuTriCub33gamma[6]*A0 + cuTriCub33alpha[6]*B0 + cuTriCub33beta[6]*C0;
    cub33Q[1] = cuTriCub33gamma[6]*A1 + cuTriCub33alpha[6]*B1 + cuTriCub33beta[6]*C1;
    cub33Q[2] = cuTriCub33gamma[6]*A2 + cuTriCub33alpha[6]*B2 + cuTriCub33beta[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[25]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [26] beta_6 A, gamma_6 B, alpha_6 C

    cub33Q[0] = cuTriCub33beta[6]*A0 + cuTriCub33gamma[6]*B0 + cuTriCub33alpha[6]*C0;
    cub33Q[1] = cuTriCub33beta[6]*A1 + cuTriCub33gamma[6]*B1 + cuTriCub33alpha[6]*C1;
    cub33Q[2] = cuTriCub33beta[6]*A2 + cuTriCub33gamma[6]*B2 + cuTriCub33alpha[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[26]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [27] alpha_7 A, beta_7 B, gamma_7 C

    cub33Q[0] = cuTriCub33alpha[7]*A0 + cuTriCub33beta[7]*B0 + cuTriCub33gamma[7]*C0;
    cub33Q[1] = cuTriCub33alpha[7]*A1 + cuTriCub33beta[7]*B1 + cuTriCub33gamma[7]*C1;
    cub33Q[2] = cuTriCub33alpha[7]*A2 + cuTriCub33beta[7]*B2 + cuTriCub33gamma[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[27]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [28] beta_7 A, alpha_7 B, gamma_7 C

    cub33Q[0] = cuTriCub33beta[7]*A0 + cuTriCub33alpha[7]*B0 + cuTriCub33gamma[7]*C0;
    cub33Q[1] = cuTriCub33beta[7]*A1 + cuTriCub33alpha[7]*B1 + cuTriCub33gamma[7]*C1;
    cub33Q[2] = cuTriCub33beta[7]*A2 + cuTriCub33alpha[7]*B2 + cuTriCub33gamma[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[28]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [29] gamma_7 A, beta_7 B, alpha_7 C

    cub33Q[0] = cuTriCub33gamma[7]*A0 + cuTriCub33beta[7]*B0 + cuTriCub33alpha[7]*C0;
    cub33Q[1] = cuTriCub33gamma[7]*A1 + cuTriCub33beta[7]*B1 + cuTriCub33alpha[7]*C1;
    cub33Q[2] = cuTriCub33gamma[7]*A2 + cuTriCub33beta[7]*B2 + cuTriCub33alpha[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[29]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [30] alpha_7 A, gamma_7 B, beta_7 C

    cub33Q[0] = cuTriCub33alpha[7]*A0 + cuTriCub33gamma[7]*B0 + cuTriCub33beta[7]*C0;
    cub33Q[1] = cuTriCub33alpha[7]*A1 + cuTriCub33gamma[7]*B1 + cuTriCub33beta[7]*C1;
    cub33Q[2] = cuTriCub33alpha[7]*A2 + cuTriCub33gamma[7]*B2 + cuTriCub33beta[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[30]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [31] gamma_7 A, alpha_7 B, beta_7 C

    cub33Q[0] = cuTriCub33gamma[7]*A0 + cuTriCub33alpha[7]*B0 + cuTriCub33beta[7]*C0;
    cub33Q[1] = cuTriCub33gamma[7]*A1 + cuTriCub33alpha[7]*B1 + cuTriCub33beta[7]*C1;
    cub33Q[2] = cuTriCub33gamma[7]*A2 + cuTriCub33alpha[7]*B2 + cuTriCub33beta[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[31]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [32] beta_7 A, gamma_7 B, alpha_7 C

    cub33Q[0] = cuTriCub33beta[7]*A0 + cuTriCub33gamma[7]*B0 + cuTriCub33alpha[7]*C0;
    cub33Q[1] = cuTriCub33beta[7]*A1 + cuTriCub33gamma[7]*B1 + cuTriCub33alpha[7]*C1;
    cub33Q[2] = cuTriCub33beta[7]*A2 + cuTriCub33gamma[7]*B2 + cuTriCub33alpha[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[32]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    return MAKECU4( prefacStatic*finalSum0, prefacStatic*finalSum1, prefacStatic*finalSum2, 0. );
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE4 ET_EFieldAndPotential_Cub33P( const CU_TYPE* P,  const CU_TYPE* data )
{
    const CU_TYPE prefacStatic = Tri_Area( data ) * M_ONEOVER_4PI_EPS0;
    CU_TYPE distVector[3] = {0., 0., 0.};
    CU_TYPE finalSum0 = 0.;
    CU_TYPE finalSum1 = 0.;
    CU_TYPE finalSum2 = 0.;
    CU_TYPE finalSum3 = 0.;

    // triangle corner points

    const CU_TYPE A0 = data[2];
    const CU_TYPE A1 = data[3];
    const CU_TYPE A2 = data[4];

    const CU_TYPE B0 = data[2] + (data[0]*data[5]);
    const CU_TYPE B1 = data[3] + (data[0]*data[6]);
    const CU_TYPE B2 = data[4] + (data[0]*data[7]);

    const CU_TYPE C0 = data[2] + (data[1]*data[8]);
    const CU_TYPE C1 = data[3] + (data[1]*data[9]);
    const CU_TYPE C2 = data[4] + (data[1]*data[10]);

    // [0] alpha_0 A, beta_0 B, gamma_0 C

    CU_TYPE cub33Q[3] = {
            cuTriCub33alpha[0]*A0 + cuTriCub33beta[0]*B0 + cuTriCub33gamma[0]*C0,
            cuTriCub33alpha[0]*A1 + cuTriCub33beta[0]*B1 + cuTriCub33gamma[0]*C1,
            cuTriCub33alpha[0]*A2 + cuTriCub33beta[0]*B2 + cuTriCub33gamma[0]*C2 };

    CU_TYPE oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    CU_TYPE prefacDynamic = cuTriCub33w[0]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[0]*oneOverAbsoluteValue);

    // [1] beta_0 A, alpha_0 B, gamma_0 C

    cub33Q[0] = cuTriCub33beta[0]*A0 + cuTriCub33alpha[0]*B0 + cuTriCub33gamma[0]*C0;
    cub33Q[1] = cuTriCub33beta[0]*A1 + cuTriCub33alpha[0]*B1 + cuTriCub33gamma[0]*C1;
    cub33Q[2] = cuTriCub33beta[0]*A2 + cuTriCub33alpha[0]*B2 + cuTriCub33gamma[0]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[1]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[1]*oneOverAbsoluteValue);

    // [2] gamma_0 A, beta_0 B, alpha_0 C

    cub33Q[0] = cuTriCub33gamma[0]*A0 + cuTriCub33beta[0]*B0 + cuTriCub33alpha[0]*C0;
    cub33Q[1] = cuTriCub33gamma[0]*A1 + cuTriCub33beta[0]*B1 + cuTriCub33alpha[0]*C1;
    cub33Q[2] = cuTriCub33gamma[0]*A2 + cuTriCub33beta[0]*B2 + cuTriCub33alpha[0]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[2]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[2]*oneOverAbsoluteValue);

    // [3] alpha_1 A, beta_1 B, gamma_1 C

    cub33Q[0] = cuTriCub33alpha[1]*A0 + cuTriCub33beta[1]*B0 + cuTriCub33gamma[1]*C0;
    cub33Q[1] = cuTriCub33alpha[1]*A1 + cuTriCub33beta[1]*B1 + cuTriCub33gamma[1]*C1;
    cub33Q[2] = cuTriCub33alpha[1]*A2 + cuTriCub33beta[1]*B2 + cuTriCub33gamma[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[3]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[3]*oneOverAbsoluteValue);

    // [4] beta_1 A, alpha_1 B, gamma_1 C

    cub33Q[0] = cuTriCub33beta[1]*A0 + cuTriCub33alpha[1]*B0 + cuTriCub33gamma[1]*C0;
    cub33Q[1] = cuTriCub33beta[1]*A1 + cuTriCub33alpha[1]*B1 + cuTriCub33gamma[1]*C1;
    cub33Q[2] = cuTriCub33beta[1]*A2 + cuTriCub33alpha[1]*B2 + cuTriCub33gamma[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[4]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[4]*oneOverAbsoluteValue);

    // [5] gamma_1 A, beta_1 B, alpha_1 C

    cub33Q[0] = cuTriCub33gamma[1]*A0 + cuTriCub33beta[1]*B0 + cuTriCub33alpha[1]*C0;
    cub33Q[1] = cuTriCub33gamma[1]*A1 + cuTriCub33beta[1]*B1 + cuTriCub33alpha[1]*C1;
    cub33Q[2] = cuTriCub33gamma[1]*A2 + cuTriCub33beta[1]*B2 + cuTriCub33alpha[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[5]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[5]*oneOverAbsoluteValue);

    // [6] alpha_2 A, beta_2 B, gamma_2 C

    cub33Q[0] = cuTriCub33alpha[2]*A0 + cuTriCub33beta[2]*B0 + cuTriCub33gamma[2]*C0;
    cub33Q[1] = cuTriCub33alpha[2]*A1 + cuTriCub33beta[2]*B1 + cuTriCub33gamma[2]*C1;
    cub33Q[2] = cuTriCub33alpha[2]*A2 + cuTriCub33beta[2]*B2 + cuTriCub33gamma[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[6]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[6]*oneOverAbsoluteValue);

    // [7] beta_2 A, alpha_2 B, gamma_2 C

    cub33Q[0] = cuTriCub33beta[2]*A0 + cuTriCub33alpha[2]*B0 + cuTriCub33gamma[2]*C0;
    cub33Q[1] = cuTriCub33beta[2]*A1 + cuTriCub33alpha[2]*B1 + cuTriCub33gamma[2]*C1;
    cub33Q[2] = cuTriCub33beta[2]*A2 + cuTriCub33alpha[2]*B2 + cuTriCub33gamma[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[7]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[7]*oneOverAbsoluteValue);

    // [8] gamma_2 A, beta_2 B, alpha_2 C

    cub33Q[0] = cuTriCub33gamma[2]*A0 + cuTriCub33beta[2]*B0 + cuTriCub33alpha[2]*C0;
    cub33Q[1] = cuTriCub33gamma[2]*A1 + cuTriCub33beta[2]*B1 + cuTriCub33alpha[2]*C1;
    cub33Q[2] = cuTriCub33gamma[2]*A2 + cuTriCub33beta[2]*B2 + cuTriCub33alpha[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[8]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[8]*oneOverAbsoluteValue);

    // [9] alpha_3 A, beta_3 B, gamma_3 C

    cub33Q[0] = cuTriCub33alpha[3]*A0 + cuTriCub33beta[3]*B0 + cuTriCub33gamma[3]*C0;
    cub33Q[1] = cuTriCub33alpha[3]*A1 + cuTriCub33beta[3]*B1 + cuTriCub33gamma[3]*C1;
    cub33Q[2] = cuTriCub33alpha[3]*A2 + cuTriCub33beta[3]*B2 + cuTriCub33gamma[3]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[9]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[9]*oneOverAbsoluteValue);

    // [10] beta_3 A, alpha_3 B, gamma_3 C

    cub33Q[0] = cuTriCub33beta[3]*A0 + cuTriCub33alpha[3]*B0 + cuTriCub33gamma[3]*C0;
    cub33Q[1] = cuTriCub33beta[3]*A1 + cuTriCub33alpha[3]*B1 + cuTriCub33gamma[3]*C1;
    cub33Q[2] = cuTriCub33beta[3]*A2 + cuTriCub33alpha[3]*B2 + cuTriCub33gamma[3]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[10]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[10]*oneOverAbsoluteValue);

    // [11] gamma_3 A, beta_3 B, alpha_3 C

    cub33Q[0] = cuTriCub33gamma[3]*A0 + cuTriCub33beta[3]*B0 + cuTriCub33alpha[3]*C0;
    cub33Q[1] = cuTriCub33gamma[3]*A1 + cuTriCub33beta[3]*B1 + cuTriCub33alpha[3]*C1;
    cub33Q[2] = cuTriCub33gamma[3]*A2 + cuTriCub33beta[3]*B2 + cuTriCub33alpha[3]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[11]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[11]*oneOverAbsoluteValue);

    // [12] alpha_4 A, beta_4 B, gamma_4 C

    cub33Q[0] = cuTriCub33alpha[4]*A0 + cuTriCub33beta[4]*B0 + cuTriCub33gamma[4]*C0;
    cub33Q[1] = cuTriCub33alpha[4]*A1 + cuTriCub33beta[4]*B1 + cuTriCub33gamma[4]*C1;
    cub33Q[2] = cuTriCub33alpha[4]*A2 + cuTriCub33beta[4]*B2 + cuTriCub33gamma[4]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[12]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[12]*oneOverAbsoluteValue);

    // [13] beta_4 A, alpha_4 B, gamma_4 C

    cub33Q[0] = cuTriCub33beta[4]*A0 + cuTriCub33alpha[4]*B0 + cuTriCub33gamma[4]*C0;
    cub33Q[1] = cuTriCub33beta[4]*A1 + cuTriCub33alpha[4]*B1 + cuTriCub33gamma[4]*C1;
    cub33Q[2] = cuTriCub33beta[4]*A2 + cuTriCub33alpha[4]*B2 + cuTriCub33gamma[4]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[13]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[13]*oneOverAbsoluteValue);

    // [14] gamma_4 A, beta_4 B, alpha_4 C

    cub33Q[0] = cuTriCub33gamma[4]*A0 + cuTriCub33beta[4]*B0 + cuTriCub33alpha[4]*C0;
    cub33Q[1] = cuTriCub33gamma[4]*A1 + cuTriCub33beta[4]*B1 + cuTriCub33alpha[4]*C1;
    cub33Q[2] = cuTriCub33gamma[4]*A2 + cuTriCub33beta[4]*B2 + cuTriCub33alpha[4]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[14]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[14]*oneOverAbsoluteValue);

    // [15] alpha_5 A, beta_5 B, gamma_5 C

    cub33Q[0] = cuTriCub33alpha[5]*A0 + cuTriCub33beta[5]*B0 + cuTriCub33gamma[5]*C0;
    cub33Q[1] = cuTriCub33alpha[5]*A1 + cuTriCub33beta[5]*B1 + cuTriCub33gamma[5]*C1;
    cub33Q[2] = cuTriCub33alpha[5]*A2 + cuTriCub33beta[5]*B2 + cuTriCub33gamma[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[15]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[15]*oneOverAbsoluteValue);

    // [16] beta_5 A, alpha_5 B, gamma_5 C

    cub33Q[0] = cuTriCub33beta[5]*A0 + cuTriCub33alpha[5]*B0 + cuTriCub33gamma[5]*C0;
    cub33Q[1] = cuTriCub33beta[5]*A1 + cuTriCub33alpha[5]*B1 + cuTriCub33gamma[5]*C1;
    cub33Q[2] = cuTriCub33beta[5]*A2 + cuTriCub33alpha[5]*B2 + cuTriCub33gamma[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[16]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[16]*oneOverAbsoluteValue);

    // [17] gamma_5 A, beta_5 B, alpha_5 C

    cub33Q[0] = cuTriCub33gamma[5]*A0 + cuTriCub33beta[5]*B0 + cuTriCub33alpha[5]*C0;
    cub33Q[1] = cuTriCub33gamma[5]*A1 + cuTriCub33beta[5]*B1 + cuTriCub33alpha[5]*C1;
    cub33Q[2] = cuTriCub33gamma[5]*A2 + cuTriCub33beta[5]*B2 + cuTriCub33alpha[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[17]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[17]*oneOverAbsoluteValue);

    // [18] alpha_5 A, gamma_5 B, beta_5 C

    cub33Q[0] = cuTriCub33alpha[5]*A0 + cuTriCub33gamma[5]*B0 + cuTriCub33beta[5]*C0;
    cub33Q[1] = cuTriCub33alpha[5]*A1 + cuTriCub33gamma[5]*B1 + cuTriCub33beta[5]*C1;
    cub33Q[2] = cuTriCub33alpha[5]*A2 + cuTriCub33gamma[5]*B2 + cuTriCub33beta[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[18]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[18]*oneOverAbsoluteValue);

    // [19] gamma_5 A, alpha_5 B, beta_5 C

    cub33Q[0] = cuTriCub33gamma[5]*A0 + cuTriCub33alpha[5]*B0 + cuTriCub33beta[5]*C0;
    cub33Q[1] = cuTriCub33gamma[5]*A1 + cuTriCub33alpha[5]*B1 + cuTriCub33beta[5]*C1;
    cub33Q[2] = cuTriCub33gamma[5]*A2 + cuTriCub33alpha[5]*B2 + cuTriCub33beta[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[19]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[19]*oneOverAbsoluteValue);

    // [20] beta_5 A, gamma_5 B, alpha_5 C

    cub33Q[0] = cuTriCub33beta[5]*A0 + cuTriCub33gamma[5]*B0 + cuTriCub33alpha[5]*C0;
    cub33Q[1] = cuTriCub33beta[5]*A1 + cuTriCub33gamma[5]*B1 + cuTriCub33alpha[5]*C1;
    cub33Q[2] = cuTriCub33beta[5]*A2 + cuTriCub33gamma[5]*B2 + cuTriCub33alpha[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[20]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[20]*oneOverAbsoluteValue);

    // [21] alpha_6 A, beta_6 B, gamma_6 C

    cub33Q[0] = cuTriCub33alpha[6]*A0 + cuTriCub33beta[6]*B0 + cuTriCub33gamma[6]*C0;
    cub33Q[1] = cuTriCub33alpha[6]*A1 + cuTriCub33beta[6]*B1 + cuTriCub33gamma[6]*C1;
    cub33Q[2] = cuTriCub33alpha[6]*A2 + cuTriCub33beta[6]*B2 + cuTriCub33gamma[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[21]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[21]*oneOverAbsoluteValue);

    // [22] beta_6 A, alpha_6 B, gamma_6 C

    cub33Q[0] = cuTriCub33beta[6]*A0 + cuTriCub33alpha[6]*B0 + cuTriCub33gamma[6]*C0;
    cub33Q[1] = cuTriCub33beta[6]*A1 + cuTriCub33alpha[6]*B1 + cuTriCub33gamma[6]*C1;
    cub33Q[2] = cuTriCub33beta[6]*A2 + cuTriCub33alpha[6]*B2 + cuTriCub33gamma[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[22]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[22]*oneOverAbsoluteValue);

    // [23] gamma_6 A, beta_6 B, alpha_6 C

    cub33Q[0] = cuTriCub33gamma[6]*A0 + cuTriCub33beta[6]*B0 + cuTriCub33alpha[6]*C0;
    cub33Q[1] = cuTriCub33gamma[6]*A1 + cuTriCub33beta[6]*B1 + cuTriCub33alpha[6]*C1;
    cub33Q[2] = cuTriCub33gamma[6]*A2 + cuTriCub33beta[6]*B2 + cuTriCub33alpha[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[23]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[23]*oneOverAbsoluteValue);

    // [24] alpha_6 A, gamma_6 B, beta_6 C

    cub33Q[0] = cuTriCub33alpha[6]*A0 + cuTriCub33gamma[6]*B0 + cuTriCub33beta[6]*C0;
    cub33Q[1] = cuTriCub33alpha[6]*A1 + cuTriCub33gamma[6]*B1 + cuTriCub33beta[6]*C1;
    cub33Q[2] = cuTriCub33alpha[6]*A2 + cuTriCub33gamma[6]*B2 + cuTriCub33beta[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[24]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[24]*oneOverAbsoluteValue);

    // [25] gamma_6 A, alpha_6 B, beta_6 C

    cub33Q[0] = cuTriCub33gamma[6]*A0 + cuTriCub33alpha[6]*B0 + cuTriCub33beta[6]*C0;
    cub33Q[1] = cuTriCub33gamma[6]*A1 + cuTriCub33alpha[6]*B1 + cuTriCub33beta[6]*C1;
    cub33Q[2] = cuTriCub33gamma[6]*A2 + cuTriCub33alpha[6]*B2 + cuTriCub33beta[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[25]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[25]*oneOverAbsoluteValue);

    // [26] beta_6 A, gamma_6 B, alpha_6 C

    cub33Q[0] = cuTriCub33beta[6]*A0 + cuTriCub33gamma[6]*B0 + cuTriCub33alpha[6]*C0;
    cub33Q[1] = cuTriCub33beta[6]*A1 + cuTriCub33gamma[6]*B1 + cuTriCub33alpha[6]*C1;
    cub33Q[2] = cuTriCub33beta[6]*A2 + cuTriCub33gamma[6]*B2 + cuTriCub33alpha[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[26]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[26]*oneOverAbsoluteValue);

    // [27] alpha_7 A, beta_7 B, gamma_7 C

    cub33Q[0] = cuTriCub33alpha[7]*A0 + cuTriCub33beta[7]*B0 + cuTriCub33gamma[7]*C0;
    cub33Q[1] = cuTriCub33alpha[7]*A1 + cuTriCub33beta[7]*B1 + cuTriCub33gamma[7]*C1;
    cub33Q[2] = cuTriCub33alpha[7]*A2 + cuTriCub33beta[7]*B2 + cuTriCub33gamma[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[27]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[27]*oneOverAbsoluteValue);

    // [28] beta_7 A, alpha_7 B, gamma_7 C

    cub33Q[0] = cuTriCub33beta[7]*A0 + cuTriCub33alpha[7]*B0 + cuTriCub33gamma[7]*C0;
    cub33Q[1] = cuTriCub33beta[7]*A1 + cuTriCub33alpha[7]*B1 + cuTriCub33gamma[7]*C1;
    cub33Q[2] = cuTriCub33beta[7]*A2 + cuTriCub33alpha[7]*B2 + cuTriCub33gamma[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[28]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[28]*oneOverAbsoluteValue);

    // [29] gamma_7 A, beta_7 B, alpha_7 C

    cub33Q[0] = cuTriCub33gamma[7]*A0 + cuTriCub33beta[7]*B0 + cuTriCub33alpha[7]*C0;
    cub33Q[1] = cuTriCub33gamma[7]*A1 + cuTriCub33beta[7]*B1 + cuTriCub33alpha[7]*C1;
    cub33Q[2] = cuTriCub33gamma[7]*A2 + cuTriCub33beta[7]*B2 + cuTriCub33alpha[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[29]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[29]*oneOverAbsoluteValue);

    // [30] alpha_7 A, gamma_7 B, beta_7 C

    cub33Q[0] = cuTriCub33alpha[7]*A0 + cuTriCub33gamma[7]*B0 + cuTriCub33beta[7]*C0;
    cub33Q[1] = cuTriCub33alpha[7]*A1 + cuTriCub33gamma[7]*B1 + cuTriCub33beta[7]*C1;
    cub33Q[2] = cuTriCub33alpha[7]*A2 + cuTriCub33gamma[7]*B2 + cuTriCub33beta[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[30]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[30]*oneOverAbsoluteValue);

    // [31] gamma_7 A, alpha_7 B, beta_7 C

    cub33Q[0] = cuTriCub33gamma[7]*A0 + cuTriCub33alpha[7]*B0 + cuTriCub33beta[7]*C0;
    cub33Q[1] = cuTriCub33gamma[7]*A1 + cuTriCub33alpha[7]*B1 + cuTriCub33beta[7]*C1;
    cub33Q[2] = cuTriCub33gamma[7]*A2 + cuTriCub33alpha[7]*B2 + cuTriCub33beta[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[31]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[31]*oneOverAbsoluteValue);

    // [32] beta_7 A, gamma_7 B, alpha_7 C

    cub33Q[0] = cuTriCub33beta[7]*A0 + cuTriCub33gamma[7]*B0 + cuTriCub33alpha[7]*C0;
    cub33Q[1] = cuTriCub33beta[7]*A1 + cuTriCub33gamma[7]*B1 + cuTriCub33alpha[7]*C1;
    cub33Q[2] = cuTriCub33beta[7]*A2 + cuTriCub33gamma[7]*B2 + cuTriCub33alpha[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
    prefacDynamic = cuTriCub33w[32]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub33w[32]*oneOverAbsoluteValue);

    return MAKECU4( prefacStatic*finalSum0, prefacStatic*finalSum1, prefacStatic*finalSum2, prefacStatic*finalSum3 );
}

#endif /* KEMFIELD_ELECTROSTATICCUBATURETRIANGLE_33POINT_CUH */
