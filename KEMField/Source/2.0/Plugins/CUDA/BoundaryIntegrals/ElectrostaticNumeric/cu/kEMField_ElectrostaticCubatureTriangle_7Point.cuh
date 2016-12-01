#ifndef KEMFIELD_ELECTROSTATICCUBATURETRIANGLE_7POINT_CUH
#define KEMFIELD_ELECTROSTATICCUBATURETRIANGLE_7POINT_CUH

// CUDA kernel for triangle boundary integrator with 7-point Gaussian cubature
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
// * ET_Potential_Cub7P: 384
// * ET_EField_Cub7P: 512
// * ET_EFieldAndPotential_Cub7P: 512


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

__constant__ CU_TYPE cuTriCub7alpha[3];
__constant__ CU_TYPE cuTriCub7beta[3];
__constant__ CU_TYPE cuTriCub7gamma[3];

// Gaussian weights

__constant__ CU_TYPE cuTriCub7w[7];

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE ET_Potential_Cub7P( const CU_TYPE* P,  const CU_TYPE* data )
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

    ///////
    // 0 //
    ///////

    /* alpha A, beta B, gamma C - 0 */

    CU_TYPE cub7Q[3] = {
    		cuTriCub7alpha[0]*A0 + cuTriCub7beta[0]*B0 + cuTriCub7gamma[0]*C0,
    		cuTriCub7alpha[0]*A1 + cuTriCub7beta[0]*B1 + cuTriCub7gamma[0]*C1,
    		cuTriCub7alpha[0]*A2 + cuTriCub7beta[0]*B2 + cuTriCub7gamma[0]*C2 };

   	finalSum += (cuTriCub7w[0]*OneOverR(cub7Q,P));

    ///////
    // 1 //
    ///////

   	/* alpha A, beta B, gamma C - 1 */

    cub7Q[0] = cuTriCub7alpha[1]*A0 + cuTriCub7beta[1]*B0 + cuTriCub7gamma[1]*C0;
    cub7Q[1] = cuTriCub7alpha[1]*A1 + cuTriCub7beta[1]*B1 + cuTriCub7gamma[1]*C1;
    cub7Q[2] = cuTriCub7alpha[1]*A2 + cuTriCub7beta[1]*B2 + cuTriCub7gamma[1]*C2;

   	finalSum += (cuTriCub7w[1]*OneOverR(cub7Q,P));

    ///////
    // 2 //
    ///////

   	/* beta A, alpha B, gamma C - 1 */

    cub7Q[0] = cuTriCub7beta[1]*A0 + cuTriCub7alpha[1]*B0 + cuTriCub7gamma[1]*C0;
    cub7Q[1] = cuTriCub7beta[1]*A1 + cuTriCub7alpha[1]*B1 + cuTriCub7gamma[1]*C1;
    cub7Q[2] = cuTriCub7beta[1]*A2 + cuTriCub7alpha[1]*B2 + cuTriCub7gamma[1]*C2;

   	finalSum += (cuTriCub7w[2]*OneOverR(cub7Q,P));

    ///////
    // 3 //
    ///////

   	/* gamma A, beta B, alpha C - 1 */

    cub7Q[0] = cuTriCub7gamma[1]*A0 + cuTriCub7beta[1]*B0 + cuTriCub7alpha[1]*C0;
    cub7Q[1] = cuTriCub7gamma[1]*A1 + cuTriCub7beta[1]*B1 + cuTriCub7alpha[1]*C1;
    cub7Q[2] = cuTriCub7gamma[1]*A2 + cuTriCub7beta[1]*B2 + cuTriCub7alpha[1]*C2;

   	finalSum += (cuTriCub7w[3]*OneOverR(cub7Q,P));

    ///////
    // 4 //
    ///////

   	/* alpha A, beta B, gamma C - 2 */

    cub7Q[0] = cuTriCub7alpha[2]*A0 + cuTriCub7beta[2]*B0 + cuTriCub7gamma[2]*C0;
    cub7Q[1] = cuTriCub7alpha[2]*A1 + cuTriCub7beta[2]*B1 + cuTriCub7gamma[2]*C1;
    cub7Q[2] = cuTriCub7alpha[2]*A2 + cuTriCub7beta[2]*B2 + cuTriCub7gamma[2]*C2;

   	finalSum += (cuTriCub7w[4]*OneOverR(cub7Q,P));

    ///////
    // 5 //
    ///////

   	/* beta A, alpha B, gamma C - 2 */

    cub7Q[0] = cuTriCub7beta[2]*A0 + cuTriCub7alpha[2]*B0 + cuTriCub7gamma[2]*C0;
    cub7Q[1] = cuTriCub7beta[2]*A1 + cuTriCub7alpha[2]*B1 + cuTriCub7gamma[2]*C1;
    cub7Q[2] = cuTriCub7beta[2]*A2 + cuTriCub7alpha[2]*B2 + cuTriCub7gamma[2]*C2;

   	finalSum += (cuTriCub7w[5]*OneOverR(cub7Q,P));

    ///////
    // 6 //
    ///////

   	/* gamma A, beta B, alpha C - 2 */

    cub7Q[0] = cuTriCub7gamma[2]*A0 + cuTriCub7beta[2]*B0 + cuTriCub7alpha[2]*C0;
    cub7Q[1] = cuTriCub7gamma[2]*A1 + cuTriCub7beta[2]*B1 + cuTriCub7alpha[2]*C1;
    cub7Q[2] = cuTriCub7gamma[2]*A2 + cuTriCub7beta[2]*B2 + cuTriCub7alpha[2]*C2;

   	finalSum += (cuTriCub7w[6]*OneOverR(cub7Q,P));

    return (prefacStatic*finalSum);
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE4 ET_EField_Cub7P( const CU_TYPE* P,  const CU_TYPE* data )
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

    ///////
    // 0 //
    ///////

    /* alpha A, beta B, gamma C - 0 */

    CU_TYPE cub7Q[3] = {
    		cuTriCub7alpha[0]*A0 + cuTriCub7beta[0]*B0 + cuTriCub7gamma[0]*C0,
    		cuTriCub7alpha[0]*A1 + cuTriCub7beta[0]*B1 + cuTriCub7gamma[0]*C1,
    		cuTriCub7alpha[0]*A2 + cuTriCub7beta[0]*B2 + cuTriCub7gamma[0]*C2 };

  	CU_TYPE oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	CU_TYPE prefacDynamic = cuTriCub7w[0]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    ///////
    // 1 //
    ///////

    /* alpha A, beta B, gamma C - 1 */

    cub7Q[0] = cuTriCub7alpha[1]*A0 + cuTriCub7beta[1]*B0 + cuTriCub7gamma[1]*C0;
    cub7Q[1] = cuTriCub7alpha[1]*A1 + cuTriCub7beta[1]*B1 + cuTriCub7gamma[1]*C1;
    cub7Q[2] = cuTriCub7alpha[1]*A2 + cuTriCub7beta[1]*B2 + cuTriCub7gamma[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = cuTriCub7w[1]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    ///////
    // 2 //
    ///////

    /* beta A, alpha B, gamma C - 1 */

    cub7Q[0] = cuTriCub7beta[1]*A0 + cuTriCub7alpha[1]*B0 + cuTriCub7gamma[1]*C0;
    cub7Q[1] = cuTriCub7beta[1]*A1 + cuTriCub7alpha[1]*B1 + cuTriCub7gamma[1]*C1;
    cub7Q[2] = cuTriCub7beta[1]*A2 + cuTriCub7alpha[1]*B2 + cuTriCub7gamma[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = cuTriCub7w[2]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    ///////
    // 3 //
    ///////

    /* gamma A, beta B, alpha C - 1 */

    cub7Q[0] = cuTriCub7gamma[1]*A0 + cuTriCub7beta[1]*B0 + cuTriCub7alpha[1]*C0;
    cub7Q[1] = cuTriCub7gamma[1]*A1 + cuTriCub7beta[1]*B1 + cuTriCub7alpha[1]*C1;
    cub7Q[2] = cuTriCub7gamma[1]*A2 + cuTriCub7beta[1]*B2 + cuTriCub7alpha[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = cuTriCub7w[3]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    ///////
    // 4 //
    ///////

    /* alpha A, beta B, gamma C - 2 */

    cub7Q[0] = cuTriCub7alpha[2]*A0 + cuTriCub7beta[2]*B0 + cuTriCub7gamma[2]*C0;
    cub7Q[1] = cuTriCub7alpha[2]*A1 + cuTriCub7beta[2]*B1 + cuTriCub7gamma[2]*C1;
    cub7Q[2] = cuTriCub7alpha[2]*A2 + cuTriCub7beta[2]*B2 + cuTriCub7gamma[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = cuTriCub7w[4]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    ///////
    // 5 //
    ///////

    /* beta A, alpha B, gamma C - 2 */

    cub7Q[0] = cuTriCub7beta[2]*A0 + cuTriCub7alpha[2]*B0 + cuTriCub7gamma[2]*C0;
    cub7Q[1] = cuTriCub7beta[2]*A1 + cuTriCub7alpha[2]*B1 + cuTriCub7gamma[2]*C1;
    cub7Q[2] = cuTriCub7beta[2]*A2 + cuTriCub7alpha[2]*B2 + cuTriCub7gamma[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = cuTriCub7w[5]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    ///////
    // 6 //
    ///////

    /* gamma A, beta B, alpha C - 2 */

    cub7Q[0] = cuTriCub7gamma[2]*A0 + cuTriCub7beta[2]*B0 + cuTriCub7alpha[2]*C0;
    cub7Q[1] = cuTriCub7gamma[2]*A1 + cuTriCub7beta[2]*B1 + cuTriCub7alpha[2]*C1;
    cub7Q[2] = cuTriCub7gamma[2]*A2 + cuTriCub7beta[2]*B2 + cuTriCub7alpha[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = cuTriCub7w[6]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    return MAKECU4( prefacStatic*finalSum0, prefacStatic*finalSum1, prefacStatic*finalSum2, 0. );
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE4 ET_EFieldAndPotential_Cub7P( const CU_TYPE* P,  const CU_TYPE* data )
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

    ///////
    // 0 //
    ///////

    /* alpha A, beta B, gamma C - 0 */

    CU_TYPE cub7Q[3] = {
    		cuTriCub7alpha[0]*A0 + cuTriCub7beta[0]*B0 + cuTriCub7gamma[0]*C0,
    		cuTriCub7alpha[0]*A1 + cuTriCub7beta[0]*B1 + cuTriCub7gamma[0]*C1,
    		cuTriCub7alpha[0]*A2 + cuTriCub7beta[0]*B2 + cuTriCub7gamma[0]*C2 };

  	CU_TYPE oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	CU_TYPE prefacDynamic = cuTriCub7w[0]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub7w[0]*oneOverAbsoluteValue);

    ///////
    // 1 //
    ///////

    /* alpha A, beta B, gamma C - 1 */

    cub7Q[0] = cuTriCub7alpha[1]*A0 + cuTriCub7beta[1]*B0 + cuTriCub7gamma[1]*C0;
    cub7Q[1] = cuTriCub7alpha[1]*A1 + cuTriCub7beta[1]*B1 + cuTriCub7gamma[1]*C1;
    cub7Q[2] = cuTriCub7alpha[1]*A2 + cuTriCub7beta[1]*B2 + cuTriCub7gamma[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = cuTriCub7w[1]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub7w[1]*oneOverAbsoluteValue);

    ///////
    // 2 //
    ///////

    /* beta A, alpha B, gamma C - 1 */

    cub7Q[0] = cuTriCub7beta[1]*A0 + cuTriCub7alpha[1]*B0 + cuTriCub7gamma[1]*C0;
    cub7Q[1] = cuTriCub7beta[1]*A1 + cuTriCub7alpha[1]*B1 + cuTriCub7gamma[1]*C1;
    cub7Q[2] = cuTriCub7beta[1]*A2 + cuTriCub7alpha[1]*B2 + cuTriCub7gamma[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = cuTriCub7w[2]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub7w[2]*oneOverAbsoluteValue);

    ///////
    // 3 //
    ///////

    /* gamma A, beta B, alpha C - 1 */

    cub7Q[0] = cuTriCub7gamma[1]*A0 + cuTriCub7beta[1]*B0 + cuTriCub7alpha[1]*C0;
    cub7Q[1] = cuTriCub7gamma[1]*A1 + cuTriCub7beta[1]*B1 + cuTriCub7alpha[1]*C1;
    cub7Q[2] = cuTriCub7gamma[1]*A2 + cuTriCub7beta[1]*B2 + cuTriCub7alpha[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = cuTriCub7w[3]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub7w[3]*oneOverAbsoluteValue);

    ///////
    // 4 //
    ///////

    /* alpha A, beta B, gamma C - 2 */

    cub7Q[0] = cuTriCub7alpha[2]*A0 + cuTriCub7beta[2]*B0 + cuTriCub7gamma[2]*C0;
    cub7Q[1] = cuTriCub7alpha[2]*A1 + cuTriCub7beta[2]*B1 + cuTriCub7gamma[2]*C1;
    cub7Q[2] = cuTriCub7alpha[2]*A2 + cuTriCub7beta[2]*B2 + cuTriCub7gamma[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = cuTriCub7w[4]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub7w[4]*oneOverAbsoluteValue);

    ///////
    // 5 //
    ///////

    /* beta A, alpha B, gamma C - 2 */

    cub7Q[0] = cuTriCub7beta[2]*A0 + cuTriCub7alpha[2]*B0 + cuTriCub7gamma[2]*C0;
    cub7Q[1] = cuTriCub7beta[2]*A1 + cuTriCub7alpha[2]*B1 + cuTriCub7gamma[2]*C1;
    cub7Q[2] = cuTriCub7beta[2]*A2 + cuTriCub7alpha[2]*B2 + cuTriCub7gamma[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = cuTriCub7w[5]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub7w[5]*oneOverAbsoluteValue);

    ///////
    // 6 //
    ///////

    /* gamma A, beta B, alpha C - 2 */

    cub7Q[0] = cuTriCub7gamma[2]*A0 + cuTriCub7beta[2]*B0 + cuTriCub7alpha[2]*C0;
    cub7Q[1] = cuTriCub7gamma[2]*A1 + cuTriCub7beta[2]*B1 + cuTriCub7alpha[2]*C1;
    cub7Q[2] = cuTriCub7gamma[2]*A2 + cuTriCub7beta[2]*B2 + cuTriCub7alpha[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = cuTriCub7w[6]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuTriCub7w[6]*oneOverAbsoluteValue);

    return MAKECU4( prefacStatic*finalSum0, prefacStatic*finalSum1, prefacStatic*finalSum2, prefacStatic*finalSum3 );
}

#endif /* KEMFIELD_ELECTROSTATICCUBATURETRIANGLE_7POINT_CUH */
