#ifndef KEMFIELD_ELECTROSTATICCUBATURETRIANGLE_7POINT_CL
#define KEMFIELD_ELECTROSTATICCUBATURETRIANGLE_7POINT_CL

// OpenCL kernel for triangle boundary integrator with 7-point Gaussian cubature
// Detailed information on the cubature implementation can be found in the CPU code,
// class 'KElectrostaticCubatureTriangleIntegrator'.
// Author: Daniel Hilk
//
// This kernel version is optimized regarding workgroup size and speed for compute
// devices providing only scalar units, but runs as well very efficient on devices
// providing vector units additionally. The optimal sizes for the used hardware
// will be set automatically.
//
// Recommended workgroup sizes by driver for NVIDIA Tesla K40c:
// * ET_Potential_Cub7P: 1024
// * ET_EField_Cub7P: 896
// * ET_EFieldAndPotential_Cub7P: 896


#include "kEMField_Triangle.cl"
#include "kEMField_ElectrostaticCubature_CommonFunctions.cl"

// Triangle geometry definition (as defined by the streamers in KTriangle.hh):
//
// data[0]:     A
// data[1]:     B
// data[2..4]:  P0[0..2]
// data[5..7]:  N1[0..2]
// data[8..10]: N2[0..2]
      
//______________________________________________________________________________

	// barycentric (area) coordinates of the Gaussian points

	__constant CL_TYPE oclTriCub7alpha[3] = {
			1./3.,
			0.0597158717897698,
			0.7974269853530873 };
	__constant CL_TYPE oclTriCub7beta[3] = {
			1./3.,
			0.470142064105115,
			0.101286507323456 };
	__constant CL_TYPE oclTriCub7gamma[3] = {
			1./3.,
			0.470142064105115,
			0.101286507323456 };

	// Gaussian weights

	__constant CL_TYPE oclTriCub7w[7] = {
			9./40.,
			0.1323941527885062,
			0.1323941527885062,
			0.1323941527885062,
			0.1259391805448272,
			0.1259391805448272,
			0.1259391805448272 };

//______________________________________________________________________________

CL_TYPE ET_Potential_Cub7P( const CL_TYPE* P, __global const CL_TYPE* data )
{
    const CL_TYPE prefacStatic = Tri_Area( data ) * M_ONEOVER_4PI_EPS0;
    CL_TYPE finalSum = 0.;

    // triangle corner points

    const CL_TYPE A0 = data[2];
    const CL_TYPE A1 = data[3];
    const CL_TYPE A2 = data[4];

    const CL_TYPE B0 = data[2] + (data[0]*data[5]);
    const CL_TYPE B1 = data[3] + (data[0]*data[6]);
    const CL_TYPE B2 = data[4] + (data[0]*data[7]);

    const CL_TYPE C0 = data[2] + (data[1]*data[8]);
    const CL_TYPE C1 = data[3] + (data[1]*data[9]);
    const CL_TYPE C2 = data[4] + (data[1]*data[10]);

    ///////
    // 0 //
    ///////

    /* alpha A, beta B, gamma C - 0 */

    CL_TYPE cub7Q[3] = {
    		oclTriCub7alpha[0]*A0 + oclTriCub7beta[0]*B0 + oclTriCub7gamma[0]*C0,
    		oclTriCub7alpha[0]*A1 + oclTriCub7beta[0]*B1 + oclTriCub7gamma[0]*C1,
    		oclTriCub7alpha[0]*A2 + oclTriCub7beta[0]*B2 + oclTriCub7gamma[0]*C2 };

   	finalSum += (oclTriCub7w[0]*OneOverR(cub7Q,P));

    ///////
    // 1 //
    ///////

   	/* alpha A, beta B, gamma C - 1 */

    cub7Q[0] = oclTriCub7alpha[1]*A0 + oclTriCub7beta[1]*B0 + oclTriCub7gamma[1]*C0;
    cub7Q[1] = oclTriCub7alpha[1]*A1 + oclTriCub7beta[1]*B1 + oclTriCub7gamma[1]*C1;
    cub7Q[2] = oclTriCub7alpha[1]*A2 + oclTriCub7beta[1]*B2 + oclTriCub7gamma[1]*C2;

   	finalSum += (oclTriCub7w[1]*OneOverR(cub7Q,P));

    ///////
    // 2 //
    ///////

   	/* beta A, alpha B, gamma C - 1 */

    cub7Q[0] = oclTriCub7beta[1]*A0 + oclTriCub7alpha[1]*B0 + oclTriCub7gamma[1]*C0;
    cub7Q[1] = oclTriCub7beta[1]*A1 + oclTriCub7alpha[1]*B1 + oclTriCub7gamma[1]*C1;
    cub7Q[2] = oclTriCub7beta[1]*A2 + oclTriCub7alpha[1]*B2 + oclTriCub7gamma[1]*C2;

   	finalSum += (oclTriCub7w[2]*OneOverR(cub7Q,P));

    ///////
    // 3 //
    ///////

   	/* gamma A, beta B, alpha C - 1 */

    cub7Q[0] = oclTriCub7gamma[1]*A0 + oclTriCub7beta[1]*B0 + oclTriCub7alpha[1]*C0;
    cub7Q[1] = oclTriCub7gamma[1]*A1 + oclTriCub7beta[1]*B1 + oclTriCub7alpha[1]*C1;
    cub7Q[2] = oclTriCub7gamma[1]*A2 + oclTriCub7beta[1]*B2 + oclTriCub7alpha[1]*C2;

   	finalSum += (oclTriCub7w[3]*OneOverR(cub7Q,P));

    ///////
    // 4 //
    ///////

   	/* alpha A, beta B, gamma C - 2 */

    cub7Q[0] = oclTriCub7alpha[2]*A0 + oclTriCub7beta[2]*B0 + oclTriCub7gamma[2]*C0;
    cub7Q[1] = oclTriCub7alpha[2]*A1 + oclTriCub7beta[2]*B1 + oclTriCub7gamma[2]*C1;
    cub7Q[2] = oclTriCub7alpha[2]*A2 + oclTriCub7beta[2]*B2 + oclTriCub7gamma[2]*C2;

   	finalSum += (oclTriCub7w[4]*OneOverR(cub7Q,P));

    ///////
    // 5 //
    ///////

   	/* beta A, alpha B, gamma C - 2 */

    cub7Q[0] = oclTriCub7beta[2]*A0 + oclTriCub7alpha[2]*B0 + oclTriCub7gamma[2]*C0;
    cub7Q[1] = oclTriCub7beta[2]*A1 + oclTriCub7alpha[2]*B1 + oclTriCub7gamma[2]*C1;
    cub7Q[2] = oclTriCub7beta[2]*A2 + oclTriCub7alpha[2]*B2 + oclTriCub7gamma[2]*C2;

   	finalSum += (oclTriCub7w[5]*OneOverR(cub7Q,P));

    ///////
    // 6 //
    ///////

   	/* gamma A, beta B, alpha C - 2 */

    cub7Q[0] = oclTriCub7gamma[2]*A0 + oclTriCub7beta[2]*B0 + oclTriCub7alpha[2]*C0;
    cub7Q[1] = oclTriCub7gamma[2]*A1 + oclTriCub7beta[2]*B1 + oclTriCub7alpha[2]*C1;
    cub7Q[2] = oclTriCub7gamma[2]*A2 + oclTriCub7beta[2]*B2 + oclTriCub7alpha[2]*C2;

   	finalSum += (oclTriCub7w[6]*OneOverR(cub7Q,P));

    return (prefacStatic*finalSum);
}

//______________________________________________________________________________

CL_TYPE4 ET_EField_Cub7P( const CL_TYPE* P, __global const CL_TYPE* data )
{
    const CL_TYPE prefacStatic = Tri_Area( data ) * M_ONEOVER_4PI_EPS0;

    CL_TYPE distVector[3] = {0., 0., 0.};
    CL_TYPE finalSum0 = 0.;
    CL_TYPE finalSum1 = 0.;
    CL_TYPE finalSum2 = 0.;

    // triangle corner points

    const CL_TYPE A0 = data[2];
    const CL_TYPE A1 = data[3];
    const CL_TYPE A2 = data[4];

    const CL_TYPE B0 = data[2] + (data[0]*data[5]);
    const CL_TYPE B1 = data[3] + (data[0]*data[6]);
    const CL_TYPE B2 = data[4] + (data[0]*data[7]);

    const CL_TYPE C0 = data[2] + (data[1]*data[8]);
    const CL_TYPE C1 = data[3] + (data[1]*data[9]);
    const CL_TYPE C2 = data[4] + (data[1]*data[10]);

    ///////
    // 0 //
    ///////

    /* alpha A, beta B, gamma C - 0 */

    CL_TYPE cub7Q[3] = {
    		oclTriCub7alpha[0]*A0 + oclTriCub7beta[0]*B0 + oclTriCub7gamma[0]*C0,
    		oclTriCub7alpha[0]*A1 + oclTriCub7beta[0]*B1 + oclTriCub7gamma[0]*C1,
    		oclTriCub7alpha[0]*A2 + oclTriCub7beta[0]*B2 + oclTriCub7gamma[0]*C2 };

  	CL_TYPE oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	CL_TYPE prefacDynamic = oclTriCub7w[0]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    ///////
    // 1 //
    ///////

    /* alpha A, beta B, gamma C - 1 */

    cub7Q[0] = oclTriCub7alpha[1]*A0 + oclTriCub7beta[1]*B0 + oclTriCub7gamma[1]*C0;
    cub7Q[1] = oclTriCub7alpha[1]*A1 + oclTriCub7beta[1]*B1 + oclTriCub7gamma[1]*C1;
    cub7Q[2] = oclTriCub7alpha[1]*A2 + oclTriCub7beta[1]*B2 + oclTriCub7gamma[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = oclTriCub7w[1]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    ///////
    // 2 //
    ///////

    /* beta A, alpha B, gamma C - 1 */

    cub7Q[0] = oclTriCub7beta[1]*A0 + oclTriCub7alpha[1]*B0 + oclTriCub7gamma[1]*C0;
    cub7Q[1] = oclTriCub7beta[1]*A1 + oclTriCub7alpha[1]*B1 + oclTriCub7gamma[1]*C1;
    cub7Q[2] = oclTriCub7beta[1]*A2 + oclTriCub7alpha[1]*B2 + oclTriCub7gamma[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = oclTriCub7w[2]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    ///////
    // 3 //
    ///////

    /* gamma A, beta B, alpha C - 1 */

    cub7Q[0] = oclTriCub7gamma[1]*A0 + oclTriCub7beta[1]*B0 + oclTriCub7alpha[1]*C0;
    cub7Q[1] = oclTriCub7gamma[1]*A1 + oclTriCub7beta[1]*B1 + oclTriCub7alpha[1]*C1;
    cub7Q[2] = oclTriCub7gamma[1]*A2 + oclTriCub7beta[1]*B2 + oclTriCub7alpha[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = oclTriCub7w[3]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    ///////
    // 4 //
    ///////

    /* alpha A, beta B, gamma C - 2 */

    cub7Q[0] = oclTriCub7alpha[2]*A0 + oclTriCub7beta[2]*B0 + oclTriCub7gamma[2]*C0;
    cub7Q[1] = oclTriCub7alpha[2]*A1 + oclTriCub7beta[2]*B1 + oclTriCub7gamma[2]*C1;
    cub7Q[2] = oclTriCub7alpha[2]*A2 + oclTriCub7beta[2]*B2 + oclTriCub7gamma[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = oclTriCub7w[4]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    ///////
    // 5 //
    ///////

    /* beta A, alpha B, gamma C - 2 */

    cub7Q[0] = oclTriCub7beta[2]*A0 + oclTriCub7alpha[2]*B0 + oclTriCub7gamma[2]*C0;
    cub7Q[1] = oclTriCub7beta[2]*A1 + oclTriCub7alpha[2]*B1 + oclTriCub7gamma[2]*C1;
    cub7Q[2] = oclTriCub7beta[2]*A2 + oclTriCub7alpha[2]*B2 + oclTriCub7gamma[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = oclTriCub7w[5]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    ///////
    // 6 //
    ///////

    /* gamma A, beta B, alpha C - 2 */

    cub7Q[0] = oclTriCub7gamma[2]*A0 + oclTriCub7beta[2]*B0 + oclTriCub7alpha[2]*C0;
    cub7Q[1] = oclTriCub7gamma[2]*A1 + oclTriCub7beta[2]*B1 + oclTriCub7alpha[2]*C1;
    cub7Q[2] = oclTriCub7gamma[2]*A2 + oclTriCub7beta[2]*B2 + oclTriCub7alpha[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = oclTriCub7w[6]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    return (CL_TYPE4)( prefacStatic*finalSum0, prefacStatic*finalSum1, prefacStatic*finalSum2, 0. );
}

//______________________________________________________________________________

CL_TYPE4 ET_EFieldAndPotential_Cub7P( const CL_TYPE* P, __global const CL_TYPE* data )
{
    const CL_TYPE prefacStatic = Tri_Area( data ) * M_ONEOVER_4PI_EPS0;

    CL_TYPE distVector[3] = {0., 0., 0.};
    CL_TYPE finalSum0 = 0.;
    CL_TYPE finalSum1 = 0.;
    CL_TYPE finalSum2 = 0.;
    CL_TYPE finalSum3 = 0.;

    // triangle corner points

    const CL_TYPE A0 = data[2];
    const CL_TYPE A1 = data[3];
    const CL_TYPE A2 = data[4];

    const CL_TYPE B0 = data[2] + (data[0]*data[5]);
    const CL_TYPE B1 = data[3] + (data[0]*data[6]);
    const CL_TYPE B2 = data[4] + (data[0]*data[7]);

    const CL_TYPE C0 = data[2] + (data[1]*data[8]);
    const CL_TYPE C1 = data[3] + (data[1]*data[9]);
    const CL_TYPE C2 = data[4] + (data[1]*data[10]);

    ///////
    // 0 //
    ///////

    /* alpha A, beta B, gamma C - 0 */

    CL_TYPE cub7Q[3] = {
    		oclTriCub7alpha[0]*A0 + oclTriCub7beta[0]*B0 + oclTriCub7gamma[0]*C0,
    		oclTriCub7alpha[0]*A1 + oclTriCub7beta[0]*B1 + oclTriCub7gamma[0]*C1,
    		oclTriCub7alpha[0]*A2 + oclTriCub7beta[0]*B2 + oclTriCub7gamma[0]*C2 };

  	CL_TYPE oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	CL_TYPE prefacDynamic = oclTriCub7w[0]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub7w[0]*oneOverAbsoluteValue);

    ///////
    // 1 //
    ///////

    /* alpha A, beta B, gamma C - 1 */

    cub7Q[0] = oclTriCub7alpha[1]*A0 + oclTriCub7beta[1]*B0 + oclTriCub7gamma[1]*C0;
    cub7Q[1] = oclTriCub7alpha[1]*A1 + oclTriCub7beta[1]*B1 + oclTriCub7gamma[1]*C1;
    cub7Q[2] = oclTriCub7alpha[1]*A2 + oclTriCub7beta[1]*B2 + oclTriCub7gamma[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = oclTriCub7w[1]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub7w[1]*oneOverAbsoluteValue);

    ///////
    // 2 //
    ///////

    /* beta A, alpha B, gamma C - 1 */

    cub7Q[0] = oclTriCub7beta[1]*A0 + oclTriCub7alpha[1]*B0 + oclTriCub7gamma[1]*C0;
    cub7Q[1] = oclTriCub7beta[1]*A1 + oclTriCub7alpha[1]*B1 + oclTriCub7gamma[1]*C1;
    cub7Q[2] = oclTriCub7beta[1]*A2 + oclTriCub7alpha[1]*B2 + oclTriCub7gamma[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = oclTriCub7w[2]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub7w[2]*oneOverAbsoluteValue);

    ///////
    // 3 //
    ///////

    /* gamma A, beta B, alpha C - 1 */

    cub7Q[0] = oclTriCub7gamma[1]*A0 + oclTriCub7beta[1]*B0 + oclTriCub7alpha[1]*C0;
    cub7Q[1] = oclTriCub7gamma[1]*A1 + oclTriCub7beta[1]*B1 + oclTriCub7alpha[1]*C1;
    cub7Q[2] = oclTriCub7gamma[1]*A2 + oclTriCub7beta[1]*B2 + oclTriCub7alpha[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = oclTriCub7w[3]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub7w[3]*oneOverAbsoluteValue);

    ///////
    // 4 //
    ///////

    /* alpha A, beta B, gamma C - 2 */

    cub7Q[0] = oclTriCub7alpha[2]*A0 + oclTriCub7beta[2]*B0 + oclTriCub7gamma[2]*C0;
    cub7Q[1] = oclTriCub7alpha[2]*A1 + oclTriCub7beta[2]*B1 + oclTriCub7gamma[2]*C1;
    cub7Q[2] = oclTriCub7alpha[2]*A2 + oclTriCub7beta[2]*B2 + oclTriCub7gamma[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = oclTriCub7w[4]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub7w[4]*oneOverAbsoluteValue);

    ///////
    // 5 //
    ///////

    /* beta A, alpha B, gamma C - 2 */

    cub7Q[0] = oclTriCub7beta[2]*A0 + oclTriCub7alpha[2]*B0 + oclTriCub7gamma[2]*C0;
    cub7Q[1] = oclTriCub7beta[2]*A1 + oclTriCub7alpha[2]*B1 + oclTriCub7gamma[2]*C1;
    cub7Q[2] = oclTriCub7beta[2]*A2 + oclTriCub7alpha[2]*B2 + oclTriCub7gamma[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = oclTriCub7w[5]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub7w[5]*oneOverAbsoluteValue);

    ///////
    // 6 //
    ///////

    /* gamma A, beta B, alpha C - 2 */

    cub7Q[0] = oclTriCub7gamma[2]*A0 + oclTriCub7beta[2]*B0 + oclTriCub7alpha[2]*C0;
    cub7Q[1] = oclTriCub7gamma[2]*A1 + oclTriCub7beta[2]*B1 + oclTriCub7alpha[2]*C1;
    cub7Q[2] = oclTriCub7gamma[2]*A2 + oclTriCub7beta[2]*B2 + oclTriCub7alpha[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub7Q,P);
  	prefacDynamic = oclTriCub7w[6]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub7w[6]*oneOverAbsoluteValue);

    return (CL_TYPE4)( prefacStatic*finalSum0, prefacStatic*finalSum1, prefacStatic*finalSum2, prefacStatic*finalSum3 );
}

#endif /* KEMFIELD_ELECTROSTATICCUBATURETRIANGLE_7POINT_CL */
