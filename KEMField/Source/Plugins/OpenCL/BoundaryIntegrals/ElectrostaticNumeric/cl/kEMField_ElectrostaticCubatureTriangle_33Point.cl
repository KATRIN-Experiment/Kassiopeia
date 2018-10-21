#ifndef KEMFIELD_ELECTROSTATICCUBATURETRIANGLE_33POINT_CL
#define KEMFIELD_ELECTROSTATICCUBATURETRIANGLE_33POINT_CL

// OpenCL kernel for triangle boundary integrator with 33-point Gaussian cubature
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
// * ET_Potential_Cub33P: 1024
// * ET_EField_Cub33P: 896
// * ET_EFieldAndPotential_Cub33P: 768

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

	__constant CL_TYPE oclTriCub33alpha[8] = {
			0.4570749859701478,
			0.1197767026828138,
			0.0235924981089169,
			0.7814843446812914,
			0.9507072731273288,
			0.1162960196779266,
			0.0230341563552671,
			0.0213824902561706 };
	__constant CL_TYPE oclTriCub33beta[8] = {
			0.2714625070149261,
			0.4401116486585931,
			0.4882037509455416,
			0.1092578276593543,
			0.0246463634363356,
			0.2554542286385173,
			0.2916556797383410,
			0.1272797172335894 };
	__constant CL_TYPE oclTriCub33gamma[8] = {
			0.2714625070149262,
			0.4401116486585931,
			0.4882037509455415,
			0.1092578276593544,
			0.0246463634363356,
			0.6282497516835561,
			0.6853101639063919,
			0.8513377925102400 };

	// Gaussian weights

	__constant CL_TYPE oclTriCub33w[33] = {
			0.0625412131959028,
			0.0625412131959028,
			0.0625412131959028,
			0.0499183349280609,
			0.0499183349280609,
			0.0499183349280609,
			0.0242668380814520,
			0.0242668380814520,
			0.0242668380814520,
			0.0284860520688775,
			0.0284860520688775,
			0.0284860520688775,
			0.0079316425099736,
			0.0079316425099736,
			0.0079316425099736,
			0.0432273636594142,
			0.0432273636594142,
			0.0432273636594142,
			0.0432273636594142,
			0.0432273636594142,
			0.0432273636594142,
			0.0217835850386076,
			0.0217835850386076,
			0.0217835850386076,
			0.0217835850386076,
			0.0217835850386076,
			0.0217835850386076,
			0.0150836775765114,
			0.0150836775765114,
			0.0150836775765114,
			0.0150836775765114,
			0.0150836775765114,
			0.0150836775765114 };


//______________________________________________________________________________

CL_TYPE ET_Potential_Cub33P( const CL_TYPE* P, __global const CL_TYPE* data )
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

    // [0] alpha_0 A, beta_0 B, gamma_0 C

    CL_TYPE cub33Q[3] = {
    		oclTriCub33alpha[0]*A0 + oclTriCub33beta[0]*B0 + oclTriCub33gamma[0]*C0,
			oclTriCub33alpha[0]*A1 + oclTriCub33beta[0]*B1 + oclTriCub33gamma[0]*C1,
			oclTriCub33alpha[0]*A2 + oclTriCub33beta[0]*B2 + oclTriCub33gamma[0]*C2 };

   	finalSum += (oclTriCub33w[0]*OneOverR(cub33Q,P));

    // [1] beta_0 A, alpha_0 B, gamma_0 C

    cub33Q[0] = oclTriCub33beta[0]*A0 + oclTriCub33alpha[0]*B0 + oclTriCub33gamma[0]*C0;
    cub33Q[1] = oclTriCub33beta[0]*A1 + oclTriCub33alpha[0]*B1 + oclTriCub33gamma[0]*C1;
    cub33Q[2] = oclTriCub33beta[0]*A2 + oclTriCub33alpha[0]*B2 + oclTriCub33gamma[0]*C2;

   	finalSum += (oclTriCub33w[1]*OneOverR(cub33Q,P));

    // [2] gamma_0 A, beta_0 B, alpha_0 C

    cub33Q[0] = oclTriCub33gamma[0]*A0 + oclTriCub33beta[0]*B0 + oclTriCub33alpha[0]*C0;
    cub33Q[1] = oclTriCub33gamma[0]*A1 + oclTriCub33beta[0]*B1 + oclTriCub33alpha[0]*C1;
    cub33Q[2] = oclTriCub33gamma[0]*A2 + oclTriCub33beta[0]*B2 + oclTriCub33alpha[0]*C2;

   	finalSum += (oclTriCub33w[2]*OneOverR(cub33Q,P));

    // [3] alpha_1 A, beta_1 B, gamma_1 C

    cub33Q[0] = oclTriCub33alpha[1]*A0 + oclTriCub33beta[1]*B0 + oclTriCub33gamma[1]*C0;
    cub33Q[1] = oclTriCub33alpha[1]*A1 + oclTriCub33beta[1]*B1 + oclTriCub33gamma[1]*C1;
    cub33Q[2] = oclTriCub33alpha[1]*A2 + oclTriCub33beta[1]*B2 + oclTriCub33gamma[1]*C2;

   	finalSum += (oclTriCub33w[3]*OneOverR(cub33Q,P));

    // [4] beta_1 A, alpha_1 B, gamma_1 C

    cub33Q[0] = oclTriCub33beta[1]*A0 + oclTriCub33alpha[1]*B0 + oclTriCub33gamma[1]*C0;
    cub33Q[1] = oclTriCub33beta[1]*A1 + oclTriCub33alpha[1]*B1 + oclTriCub33gamma[1]*C1;
    cub33Q[2] = oclTriCub33beta[1]*A2 + oclTriCub33alpha[1]*B2 + oclTriCub33gamma[1]*C2;

   	finalSum += (oclTriCub33w[4]*OneOverR(cub33Q,P));

    // [5] gamma_1 A, beta_1 B, alpha_1 C

    cub33Q[0] = oclTriCub33gamma[1]*A0 + oclTriCub33beta[1]*B0 + oclTriCub33alpha[1]*C0;
    cub33Q[1] = oclTriCub33gamma[1]*A1 + oclTriCub33beta[1]*B1 + oclTriCub33alpha[1]*C1;
    cub33Q[2] = oclTriCub33gamma[1]*A2 + oclTriCub33beta[1]*B2 + oclTriCub33alpha[1]*C2;

   	finalSum += (oclTriCub33w[5]*OneOverR(cub33Q,P));

    // [6] alpha_2 A, beta_2 B, gamma_2 C

    cub33Q[0] = oclTriCub33alpha[2]*A0 + oclTriCub33beta[2]*B0 + oclTriCub33gamma[2]*C0;
    cub33Q[1] = oclTriCub33alpha[2]*A1 + oclTriCub33beta[2]*B1 + oclTriCub33gamma[2]*C1;
    cub33Q[2] = oclTriCub33alpha[2]*A2 + oclTriCub33beta[2]*B2 + oclTriCub33gamma[2]*C2;

   	finalSum += (oclTriCub33w[6]*OneOverR(cub33Q,P));

    // [7] beta_2 A, alpha_2 B, gamma_2 C

    cub33Q[0] = oclTriCub33beta[2]*A0 + oclTriCub33alpha[2]*B0 + oclTriCub33gamma[2]*C0;
    cub33Q[1] = oclTriCub33beta[2]*A1 + oclTriCub33alpha[2]*B1 + oclTriCub33gamma[2]*C1;
    cub33Q[2] = oclTriCub33beta[2]*A2 + oclTriCub33alpha[2]*B2 + oclTriCub33gamma[2]*C2;

   	finalSum += (oclTriCub33w[7]*OneOverR(cub33Q,P));

    // [8] gamma_2 A, beta_2 B, alpha_2 C

    cub33Q[0] = oclTriCub33gamma[2]*A0 + oclTriCub33beta[2]*B0 + oclTriCub33alpha[2]*C0;
    cub33Q[1] = oclTriCub33gamma[2]*A1 + oclTriCub33beta[2]*B1 + oclTriCub33alpha[2]*C1;
    cub33Q[2] = oclTriCub33gamma[2]*A2 + oclTriCub33beta[2]*B2 + oclTriCub33alpha[2]*C2;

   	finalSum += (oclTriCub33w[8]*OneOverR(cub33Q,P));

    // [9] alpha_3 A, beta_3 B, gamma_3 C

    cub33Q[0] = oclTriCub33alpha[3]*A0 + oclTriCub33beta[3]*B0 + oclTriCub33gamma[3]*C0;
    cub33Q[1] = oclTriCub33alpha[3]*A1 + oclTriCub33beta[3]*B1 + oclTriCub33gamma[3]*C1;
    cub33Q[2] = oclTriCub33alpha[3]*A2 + oclTriCub33beta[3]*B2 + oclTriCub33gamma[3]*C2;

   	finalSum += (oclTriCub33w[9]*OneOverR(cub33Q,P));

    // [10] beta_3 A, alpha_3 B, gamma_3 C

    cub33Q[0] = oclTriCub33beta[3]*A0 + oclTriCub33alpha[3]*B0 + oclTriCub33gamma[3]*C0;
    cub33Q[1] = oclTriCub33beta[3]*A1 + oclTriCub33alpha[3]*B1 + oclTriCub33gamma[3]*C1;
    cub33Q[2] = oclTriCub33beta[3]*A2 + oclTriCub33alpha[3]*B2 + oclTriCub33gamma[3]*C2;

   	finalSum += (oclTriCub33w[10]*OneOverR(cub33Q,P));

    // [11] gamma_3 A, beta_3 B, alpha_3 C

    cub33Q[0] = oclTriCub33gamma[3]*A0 + oclTriCub33beta[3]*B0 + oclTriCub33alpha[3]*C0;
    cub33Q[1] = oclTriCub33gamma[3]*A1 + oclTriCub33beta[3]*B1 + oclTriCub33alpha[3]*C1;
    cub33Q[2] = oclTriCub33gamma[3]*A2 + oclTriCub33beta[3]*B2 + oclTriCub33alpha[3]*C2;

   	finalSum += (oclTriCub33w[11]*OneOverR(cub33Q,P));

    // [12] alpha_4 A, beta_4 B, gamma_4 C

    cub33Q[0] = oclTriCub33alpha[4]*A0 + oclTriCub33beta[4]*B0 + oclTriCub33gamma[4]*C0;
    cub33Q[1] = oclTriCub33alpha[4]*A1 + oclTriCub33beta[4]*B1 + oclTriCub33gamma[4]*C1;
    cub33Q[2] = oclTriCub33alpha[4]*A2 + oclTriCub33beta[4]*B2 + oclTriCub33gamma[4]*C2;

   	finalSum += (oclTriCub33w[12]*OneOverR(cub33Q,P));

    // [13] beta_4 A, alpha_4 B, gamma_4 C

    cub33Q[0] = oclTriCub33beta[4]*A0 + oclTriCub33alpha[4]*B0 + oclTriCub33gamma[4]*C0;
    cub33Q[1] = oclTriCub33beta[4]*A1 + oclTriCub33alpha[4]*B1 + oclTriCub33gamma[4]*C1;
    cub33Q[2] = oclTriCub33beta[4]*A2 + oclTriCub33alpha[4]*B2 + oclTriCub33gamma[4]*C2;

   	finalSum += (oclTriCub33w[13]*OneOverR(cub33Q,P));

    // [14] gamma_4 A, beta_4 B, alpha_4 C

    cub33Q[0] = oclTriCub33gamma[4]*A0 + oclTriCub33beta[4]*B0 + oclTriCub33alpha[4]*C0;
    cub33Q[1] = oclTriCub33gamma[4]*A1 + oclTriCub33beta[4]*B1 + oclTriCub33alpha[4]*C1;
    cub33Q[2] = oclTriCub33gamma[4]*A2 + oclTriCub33beta[4]*B2 + oclTriCub33alpha[4]*C2;

   	finalSum += (oclTriCub33w[14]*OneOverR(cub33Q,P));

    // [15] alpha_5 A, beta_5 B, gamma_5 C

    cub33Q[0] = oclTriCub33alpha[5]*A0 + oclTriCub33beta[5]*B0 + oclTriCub33gamma[5]*C0;
    cub33Q[1] = oclTriCub33alpha[5]*A1 + oclTriCub33beta[5]*B1 + oclTriCub33gamma[5]*C1;
    cub33Q[2] = oclTriCub33alpha[5]*A2 + oclTriCub33beta[5]*B2 + oclTriCub33gamma[5]*C2;

   	finalSum += (oclTriCub33w[15]*OneOverR(cub33Q,P));

    // [16] beta_5 A, alpha_5 B, gamma_5 C

    cub33Q[0] = oclTriCub33beta[5]*A0 + oclTriCub33alpha[5]*B0 + oclTriCub33gamma[5]*C0;
    cub33Q[1] = oclTriCub33beta[5]*A1 + oclTriCub33alpha[5]*B1 + oclTriCub33gamma[5]*C1;
    cub33Q[2] = oclTriCub33beta[5]*A2 + oclTriCub33alpha[5]*B2 + oclTriCub33gamma[5]*C2;

   	finalSum += (oclTriCub33w[16]*OneOverR(cub33Q,P));

    // [17] gamma_5 A, beta_5 B, alpha_5 C

    cub33Q[0] = oclTriCub33gamma[5]*A0 + oclTriCub33beta[5]*B0 + oclTriCub33alpha[5]*C0;
    cub33Q[1] = oclTriCub33gamma[5]*A1 + oclTriCub33beta[5]*B1 + oclTriCub33alpha[5]*C1;
    cub33Q[2] = oclTriCub33gamma[5]*A2 + oclTriCub33beta[5]*B2 + oclTriCub33alpha[5]*C2;

   	finalSum += (oclTriCub33w[17]*OneOverR(cub33Q,P));

    // [18] alpha_5 A, gamma_5 B, beta_5 C

    cub33Q[0] = oclTriCub33alpha[5]*A0 + oclTriCub33gamma[5]*B0 + oclTriCub33beta[5]*C0;
    cub33Q[1] = oclTriCub33alpha[5]*A1 + oclTriCub33gamma[5]*B1 + oclTriCub33beta[5]*C1;
    cub33Q[2] = oclTriCub33alpha[5]*A2 + oclTriCub33gamma[5]*B2 + oclTriCub33beta[5]*C2;

   	finalSum += (oclTriCub33w[18]*OneOverR(cub33Q,P));

    // [19] gamma_5 A, alpha_5 B, beta_5 C

    cub33Q[0] = oclTriCub33gamma[5]*A0 + oclTriCub33alpha[5]*B0 + oclTriCub33beta[5]*C0;
    cub33Q[1] = oclTriCub33gamma[5]*A1 + oclTriCub33alpha[5]*B1 + oclTriCub33beta[5]*C1;
    cub33Q[2] = oclTriCub33gamma[5]*A2 + oclTriCub33alpha[5]*B2 + oclTriCub33beta[5]*C2;

   	finalSum += (oclTriCub33w[19]*OneOverR(cub33Q,P));

    // [20] beta_5 A, gamma_5 B, alpha_5 C

    cub33Q[0] = oclTriCub33beta[5]*A0 + oclTriCub33gamma[5]*B0 + oclTriCub33alpha[5]*C0;
    cub33Q[1] = oclTriCub33beta[5]*A1 + oclTriCub33gamma[5]*B1 + oclTriCub33alpha[5]*C1;
    cub33Q[2] = oclTriCub33beta[5]*A2 + oclTriCub33gamma[5]*B2 + oclTriCub33alpha[5]*C2;

   	finalSum += (oclTriCub33w[20]*OneOverR(cub33Q,P));

    // [21] alpha_6 A, beta_6 B, gamma_6 C

    cub33Q[0] = oclTriCub33alpha[6]*A0 + oclTriCub33beta[6]*B0 + oclTriCub33gamma[6]*C0;
    cub33Q[1] = oclTriCub33alpha[6]*A1 + oclTriCub33beta[6]*B1 + oclTriCub33gamma[6]*C1;
    cub33Q[2] = oclTriCub33alpha[6]*A2 + oclTriCub33beta[6]*B2 + oclTriCub33gamma[6]*C2;

   	finalSum += (oclTriCub33w[21]*OneOverR(cub33Q,P));

    // [22] beta_6 A, alpha_6 B, gamma_6 C

    cub33Q[0] = oclTriCub33beta[6]*A0 + oclTriCub33alpha[6]*B0 + oclTriCub33gamma[6]*C0;
    cub33Q[1] = oclTriCub33beta[6]*A1 + oclTriCub33alpha[6]*B1 + oclTriCub33gamma[6]*C1;
    cub33Q[2] = oclTriCub33beta[6]*A2 + oclTriCub33alpha[6]*B2 + oclTriCub33gamma[6]*C2;

   	finalSum += (oclTriCub33w[22]*OneOverR(cub33Q,P));

    // [23] gamma_6 A, beta_6 B, alpha_6 C

    cub33Q[0] = oclTriCub33gamma[6]*A0 + oclTriCub33beta[6]*B0 + oclTriCub33alpha[6]*C0;
    cub33Q[1] = oclTriCub33gamma[6]*A1 + oclTriCub33beta[6]*B1 + oclTriCub33alpha[6]*C1;
    cub33Q[2] = oclTriCub33gamma[6]*A2 + oclTriCub33beta[6]*B2 + oclTriCub33alpha[6]*C2;

   	finalSum += (oclTriCub33w[23]*OneOverR(cub33Q,P));

    // [24] alpha_6 A, gamma_6 B, beta_6 C

    cub33Q[0] = oclTriCub33alpha[6]*A0 + oclTriCub33gamma[6]*B0 + oclTriCub33beta[6]*C0;
    cub33Q[1] = oclTriCub33alpha[6]*A1 + oclTriCub33gamma[6]*B1 + oclTriCub33beta[6]*C1;
    cub33Q[2] = oclTriCub33alpha[6]*A2 + oclTriCub33gamma[6]*B2 + oclTriCub33beta[6]*C2;

   	finalSum += (oclTriCub33w[24]*OneOverR(cub33Q,P));

    // [25] gamma_6 A, alpha_6 B, beta_6 C

    cub33Q[0] = oclTriCub33gamma[6]*A0 + oclTriCub33alpha[6]*B0 + oclTriCub33beta[6]*C0;
    cub33Q[1] = oclTriCub33gamma[6]*A1 + oclTriCub33alpha[6]*B1 + oclTriCub33beta[6]*C1;
    cub33Q[2] = oclTriCub33gamma[6]*A2 + oclTriCub33alpha[6]*B2 + oclTriCub33beta[6]*C2;

   	finalSum += (oclTriCub33w[25]*OneOverR(cub33Q,P));

    // [26] beta_6 A, gamma_6 B, alpha_6 C

    cub33Q[0] = oclTriCub33beta[6]*A0 + oclTriCub33gamma[6]*B0 + oclTriCub33alpha[6]*C0;
    cub33Q[1] = oclTriCub33beta[6]*A1 + oclTriCub33gamma[6]*B1 + oclTriCub33alpha[6]*C1;
    cub33Q[2] = oclTriCub33beta[6]*A2 + oclTriCub33gamma[6]*B2 + oclTriCub33alpha[6]*C2;

   	finalSum += (oclTriCub33w[26]*OneOverR(cub33Q,P));

    // [27] alpha_7 A, beta_7 B, gamma_7 C

    cub33Q[0] = oclTriCub33alpha[7]*A0 + oclTriCub33beta[7]*B0 + oclTriCub33gamma[7]*C0;
    cub33Q[1] = oclTriCub33alpha[7]*A1 + oclTriCub33beta[7]*B1 + oclTriCub33gamma[7]*C1;
    cub33Q[2] = oclTriCub33alpha[7]*A2 + oclTriCub33beta[7]*B2 + oclTriCub33gamma[7]*C2;

   	finalSum += (oclTriCub33w[27]*OneOverR(cub33Q,P));

    // [28] beta_7 A, alpha_7 B, gamma_7 C

    cub33Q[0] = oclTriCub33beta[7]*A0 + oclTriCub33alpha[7]*B0 + oclTriCub33gamma[7]*C0;
    cub33Q[1] = oclTriCub33beta[7]*A1 + oclTriCub33alpha[7]*B1 + oclTriCub33gamma[7]*C1;
    cub33Q[2] = oclTriCub33beta[7]*A2 + oclTriCub33alpha[7]*B2 + oclTriCub33gamma[7]*C2;

   	finalSum += (oclTriCub33w[28]*OneOverR(cub33Q,P));

    // [29] gamma_7 A, beta_7 B, alpha_7 C

    cub33Q[0] = oclTriCub33gamma[7]*A0 + oclTriCub33beta[7]*B0 + oclTriCub33alpha[7]*C0;
    cub33Q[1] = oclTriCub33gamma[7]*A1 + oclTriCub33beta[7]*B1 + oclTriCub33alpha[7]*C1;
    cub33Q[2] = oclTriCub33gamma[7]*A2 + oclTriCub33beta[7]*B2 + oclTriCub33alpha[7]*C2;

   	finalSum += (oclTriCub33w[29]*OneOverR(cub33Q,P));

    // [30] alpha_7 A, gamma_7 B, beta_7 C

    cub33Q[0] = oclTriCub33alpha[7]*A0 + oclTriCub33gamma[7]*B0 + oclTriCub33beta[7]*C0;
    cub33Q[1] = oclTriCub33alpha[7]*A1 + oclTriCub33gamma[7]*B1 + oclTriCub33beta[7]*C1;
    cub33Q[2] = oclTriCub33alpha[7]*A2 + oclTriCub33gamma[7]*B2 + oclTriCub33beta[7]*C2;

   	finalSum += (oclTriCub33w[30]*OneOverR(cub33Q,P));

    // [31] gamma_7 A, alpha_7 B, beta_7 C

    cub33Q[0] = oclTriCub33gamma[7]*A0 + oclTriCub33alpha[7]*B0 + oclTriCub33beta[7]*C0;
    cub33Q[1] = oclTriCub33gamma[7]*A1 + oclTriCub33alpha[7]*B1 + oclTriCub33beta[7]*C1;
    cub33Q[2] = oclTriCub33gamma[7]*A2 + oclTriCub33alpha[7]*B2 + oclTriCub33beta[7]*C2;

   	finalSum += (oclTriCub33w[31]*OneOverR(cub33Q,P));

    // [32] beta_7 A, gamma_7 B, alpha_7 C

    cub33Q[0] = oclTriCub33beta[7]*A0 + oclTriCub33gamma[7]*B0 + oclTriCub33alpha[7]*C0;
    cub33Q[1] = oclTriCub33beta[7]*A1 + oclTriCub33gamma[7]*B1 + oclTriCub33alpha[7]*C1;
    cub33Q[2] = oclTriCub33beta[7]*A2 + oclTriCub33gamma[7]*B2 + oclTriCub33alpha[7]*C2;

   	finalSum += (oclTriCub33w[32]*OneOverR(cub33Q,P));

    return (prefacStatic*finalSum);
}

//______________________________________________________________________________

CL_TYPE4 ET_EField_Cub33P( const CL_TYPE* P, __global const CL_TYPE* data )
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

    // [0] alpha_0 A, beta_0 B, gamma_0 C

    CL_TYPE cub33Q[3] = {
    		oclTriCub33alpha[0]*A0 + oclTriCub33beta[0]*B0 + oclTriCub33gamma[0]*C0,
			oclTriCub33alpha[0]*A1 + oclTriCub33beta[0]*B1 + oclTriCub33gamma[0]*C1,
			oclTriCub33alpha[0]*A2 + oclTriCub33beta[0]*B2 + oclTriCub33gamma[0]*C2 };

  	CL_TYPE oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	CL_TYPE prefacDynamic = oclTriCub33w[0]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [1] beta_0 A, alpha_0 B, gamma_0 C

    cub33Q[0] = oclTriCub33beta[0]*A0 + oclTriCub33alpha[0]*B0 + oclTriCub33gamma[0]*C0;
    cub33Q[1] = oclTriCub33beta[0]*A1 + oclTriCub33alpha[0]*B1 + oclTriCub33gamma[0]*C1;
    cub33Q[2] = oclTriCub33beta[0]*A2 + oclTriCub33alpha[0]*B2 + oclTriCub33gamma[0]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[1]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [2] gamma_0 A, beta_0 B, alpha_0 C

    cub33Q[0] = oclTriCub33gamma[0]*A0 + oclTriCub33beta[0]*B0 + oclTriCub33alpha[0]*C0;
    cub33Q[1] = oclTriCub33gamma[0]*A1 + oclTriCub33beta[0]*B1 + oclTriCub33alpha[0]*C1;
    cub33Q[2] = oclTriCub33gamma[0]*A2 + oclTriCub33beta[0]*B2 + oclTriCub33alpha[0]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[2]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [3] alpha_1 A, beta_1 B, gamma_1 C

    cub33Q[0] = oclTriCub33alpha[1]*A0 + oclTriCub33beta[1]*B0 + oclTriCub33gamma[1]*C0;
    cub33Q[1] = oclTriCub33alpha[1]*A1 + oclTriCub33beta[1]*B1 + oclTriCub33gamma[1]*C1;
    cub33Q[2] = oclTriCub33alpha[1]*A2 + oclTriCub33beta[1]*B2 + oclTriCub33gamma[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[3]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [4] beta_1 A, alpha_1 B, gamma_1 C

    cub33Q[0] = oclTriCub33beta[1]*A0 + oclTriCub33alpha[1]*B0 + oclTriCub33gamma[1]*C0;
    cub33Q[1] = oclTriCub33beta[1]*A1 + oclTriCub33alpha[1]*B1 + oclTriCub33gamma[1]*C1;
    cub33Q[2] = oclTriCub33beta[1]*A2 + oclTriCub33alpha[1]*B2 + oclTriCub33gamma[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[4]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [5] gamma_1 A, beta_1 B, alpha_1 C

    cub33Q[0] = oclTriCub33gamma[1]*A0 + oclTriCub33beta[1]*B0 + oclTriCub33alpha[1]*C0;
    cub33Q[1] = oclTriCub33gamma[1]*A1 + oclTriCub33beta[1]*B1 + oclTriCub33alpha[1]*C1;
    cub33Q[2] = oclTriCub33gamma[1]*A2 + oclTriCub33beta[1]*B2 + oclTriCub33alpha[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[5]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [6] alpha_2 A, beta_2 B, gamma_2 C

    cub33Q[0] = oclTriCub33alpha[2]*A0 + oclTriCub33beta[2]*B0 + oclTriCub33gamma[2]*C0;
    cub33Q[1] = oclTriCub33alpha[2]*A1 + oclTriCub33beta[2]*B1 + oclTriCub33gamma[2]*C1;
    cub33Q[2] = oclTriCub33alpha[2]*A2 + oclTriCub33beta[2]*B2 + oclTriCub33gamma[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[6]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [7] beta_2 A, alpha_2 B, gamma_2 C

    cub33Q[0] = oclTriCub33beta[2]*A0 + oclTriCub33alpha[2]*B0 + oclTriCub33gamma[2]*C0;
    cub33Q[1] = oclTriCub33beta[2]*A1 + oclTriCub33alpha[2]*B1 + oclTriCub33gamma[2]*C1;
    cub33Q[2] = oclTriCub33beta[2]*A2 + oclTriCub33alpha[2]*B2 + oclTriCub33gamma[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[7]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [8] gamma_2 A, beta_2 B, alpha_2 C

    cub33Q[0] = oclTriCub33gamma[2]*A0 + oclTriCub33beta[2]*B0 + oclTriCub33alpha[2]*C0;
    cub33Q[1] = oclTriCub33gamma[2]*A1 + oclTriCub33beta[2]*B1 + oclTriCub33alpha[2]*C1;
    cub33Q[2] = oclTriCub33gamma[2]*A2 + oclTriCub33beta[2]*B2 + oclTriCub33alpha[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[8]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [9] alpha_3 A, beta_3 B, gamma_3 C

    cub33Q[0] = oclTriCub33alpha[3]*A0 + oclTriCub33beta[3]*B0 + oclTriCub33gamma[3]*C0;
    cub33Q[1] = oclTriCub33alpha[3]*A1 + oclTriCub33beta[3]*B1 + oclTriCub33gamma[3]*C1;
    cub33Q[2] = oclTriCub33alpha[3]*A2 + oclTriCub33beta[3]*B2 + oclTriCub33gamma[3]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[9]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [10] beta_3 A, alpha_3 B, gamma_3 C

    cub33Q[0] = oclTriCub33beta[3]*A0 + oclTriCub33alpha[3]*B0 + oclTriCub33gamma[3]*C0;
    cub33Q[1] = oclTriCub33beta[3]*A1 + oclTriCub33alpha[3]*B1 + oclTriCub33gamma[3]*C1;
    cub33Q[2] = oclTriCub33beta[3]*A2 + oclTriCub33alpha[3]*B2 + oclTriCub33gamma[3]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[10]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [11] gamma_3 A, beta_3 B, alpha_3 C

    cub33Q[0] = oclTriCub33gamma[3]*A0 + oclTriCub33beta[3]*B0 + oclTriCub33alpha[3]*C0;
    cub33Q[1] = oclTriCub33gamma[3]*A1 + oclTriCub33beta[3]*B1 + oclTriCub33alpha[3]*C1;
    cub33Q[2] = oclTriCub33gamma[3]*A2 + oclTriCub33beta[3]*B2 + oclTriCub33alpha[3]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[11]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [12] alpha_4 A, beta_4 B, gamma_4 C

    cub33Q[0] = oclTriCub33alpha[4]*A0 + oclTriCub33beta[4]*B0 + oclTriCub33gamma[4]*C0;
    cub33Q[1] = oclTriCub33alpha[4]*A1 + oclTriCub33beta[4]*B1 + oclTriCub33gamma[4]*C1;
    cub33Q[2] = oclTriCub33alpha[4]*A2 + oclTriCub33beta[4]*B2 + oclTriCub33gamma[4]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[12]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [13] beta_4 A, alpha_4 B, gamma_4 C

    cub33Q[0] = oclTriCub33beta[4]*A0 + oclTriCub33alpha[4]*B0 + oclTriCub33gamma[4]*C0;
    cub33Q[1] = oclTriCub33beta[4]*A1 + oclTriCub33alpha[4]*B1 + oclTriCub33gamma[4]*C1;
    cub33Q[2] = oclTriCub33beta[4]*A2 + oclTriCub33alpha[4]*B2 + oclTriCub33gamma[4]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[13]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [14] gamma_4 A, beta_4 B, alpha_4 C

    cub33Q[0] = oclTriCub33gamma[4]*A0 + oclTriCub33beta[4]*B0 + oclTriCub33alpha[4]*C0;
    cub33Q[1] = oclTriCub33gamma[4]*A1 + oclTriCub33beta[4]*B1 + oclTriCub33alpha[4]*C1;
    cub33Q[2] = oclTriCub33gamma[4]*A2 + oclTriCub33beta[4]*B2 + oclTriCub33alpha[4]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[14]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [15] alpha_5 A, beta_5 B, gamma_5 C

    cub33Q[0] = oclTriCub33alpha[5]*A0 + oclTriCub33beta[5]*B0 + oclTriCub33gamma[5]*C0;
    cub33Q[1] = oclTriCub33alpha[5]*A1 + oclTriCub33beta[5]*B1 + oclTriCub33gamma[5]*C1;
    cub33Q[2] = oclTriCub33alpha[5]*A2 + oclTriCub33beta[5]*B2 + oclTriCub33gamma[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[15]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [16] beta_5 A, alpha_5 B, gamma_5 C

    cub33Q[0] = oclTriCub33beta[5]*A0 + oclTriCub33alpha[5]*B0 + oclTriCub33gamma[5]*C0;
    cub33Q[1] = oclTriCub33beta[5]*A1 + oclTriCub33alpha[5]*B1 + oclTriCub33gamma[5]*C1;
    cub33Q[2] = oclTriCub33beta[5]*A2 + oclTriCub33alpha[5]*B2 + oclTriCub33gamma[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[16]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [17] gamma_5 A, beta_5 B, alpha_5 C

    cub33Q[0] = oclTriCub33gamma[5]*A0 + oclTriCub33beta[5]*B0 + oclTriCub33alpha[5]*C0;
    cub33Q[1] = oclTriCub33gamma[5]*A1 + oclTriCub33beta[5]*B1 + oclTriCub33alpha[5]*C1;
    cub33Q[2] = oclTriCub33gamma[5]*A2 + oclTriCub33beta[5]*B2 + oclTriCub33alpha[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[17]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [18] alpha_5 A, gamma_5 B, beta_5 C

    cub33Q[0] = oclTriCub33alpha[5]*A0 + oclTriCub33gamma[5]*B0 + oclTriCub33beta[5]*C0;
    cub33Q[1] = oclTriCub33alpha[5]*A1 + oclTriCub33gamma[5]*B1 + oclTriCub33beta[5]*C1;
    cub33Q[2] = oclTriCub33alpha[5]*A2 + oclTriCub33gamma[5]*B2 + oclTriCub33beta[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[18]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [19] gamma_5 A, alpha_5 B, beta_5 C

    cub33Q[0] = oclTriCub33gamma[5]*A0 + oclTriCub33alpha[5]*B0 + oclTriCub33beta[5]*C0;
    cub33Q[1] = oclTriCub33gamma[5]*A1 + oclTriCub33alpha[5]*B1 + oclTriCub33beta[5]*C1;
    cub33Q[2] = oclTriCub33gamma[5]*A2 + oclTriCub33alpha[5]*B2 + oclTriCub33beta[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[19]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [20] beta_5 A, gamma_5 B, alpha_5 C

    cub33Q[0] = oclTriCub33beta[5]*A0 + oclTriCub33gamma[5]*B0 + oclTriCub33alpha[5]*C0;
    cub33Q[1] = oclTriCub33beta[5]*A1 + oclTriCub33gamma[5]*B1 + oclTriCub33alpha[5]*C1;
    cub33Q[2] = oclTriCub33beta[5]*A2 + oclTriCub33gamma[5]*B2 + oclTriCub33alpha[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[20]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [21] alpha_6 A, beta_6 B, gamma_6 C

    cub33Q[0] = oclTriCub33alpha[6]*A0 + oclTriCub33beta[6]*B0 + oclTriCub33gamma[6]*C0;
    cub33Q[1] = oclTriCub33alpha[6]*A1 + oclTriCub33beta[6]*B1 + oclTriCub33gamma[6]*C1;
    cub33Q[2] = oclTriCub33alpha[6]*A2 + oclTriCub33beta[6]*B2 + oclTriCub33gamma[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[21]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [22] beta_6 A, alpha_6 B, gamma_6 C

    cub33Q[0] = oclTriCub33beta[6]*A0 + oclTriCub33alpha[6]*B0 + oclTriCub33gamma[6]*C0;
    cub33Q[1] = oclTriCub33beta[6]*A1 + oclTriCub33alpha[6]*B1 + oclTriCub33gamma[6]*C1;
    cub33Q[2] = oclTriCub33beta[6]*A2 + oclTriCub33alpha[6]*B2 + oclTriCub33gamma[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[22]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [23] gamma_6 A, beta_6 B, alpha_6 C

    cub33Q[0] = oclTriCub33gamma[6]*A0 + oclTriCub33beta[6]*B0 + oclTriCub33alpha[6]*C0;
    cub33Q[1] = oclTriCub33gamma[6]*A1 + oclTriCub33beta[6]*B1 + oclTriCub33alpha[6]*C1;
    cub33Q[2] = oclTriCub33gamma[6]*A2 + oclTriCub33beta[6]*B2 + oclTriCub33alpha[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[23]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [24] alpha_6 A, gamma_6 B, beta_6 C

    cub33Q[0] = oclTriCub33alpha[6]*A0 + oclTriCub33gamma[6]*B0 + oclTriCub33beta[6]*C0;
    cub33Q[1] = oclTriCub33alpha[6]*A1 + oclTriCub33gamma[6]*B1 + oclTriCub33beta[6]*C1;
    cub33Q[2] = oclTriCub33alpha[6]*A2 + oclTriCub33gamma[6]*B2 + oclTriCub33beta[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[24]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [25] gamma_6 A, alpha_6 B, beta_6 C

    cub33Q[0] = oclTriCub33gamma[6]*A0 + oclTriCub33alpha[6]*B0 + oclTriCub33beta[6]*C0;
    cub33Q[1] = oclTriCub33gamma[6]*A1 + oclTriCub33alpha[6]*B1 + oclTriCub33beta[6]*C1;
    cub33Q[2] = oclTriCub33gamma[6]*A2 + oclTriCub33alpha[6]*B2 + oclTriCub33beta[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[25]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [26] beta_6 A, gamma_6 B, alpha_6 C

    cub33Q[0] = oclTriCub33beta[6]*A0 + oclTriCub33gamma[6]*B0 + oclTriCub33alpha[6]*C0;
    cub33Q[1] = oclTriCub33beta[6]*A1 + oclTriCub33gamma[6]*B1 + oclTriCub33alpha[6]*C1;
    cub33Q[2] = oclTriCub33beta[6]*A2 + oclTriCub33gamma[6]*B2 + oclTriCub33alpha[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[26]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [27] alpha_7 A, beta_7 B, gamma_7 C

    cub33Q[0] = oclTriCub33alpha[7]*A0 + oclTriCub33beta[7]*B0 + oclTriCub33gamma[7]*C0;
    cub33Q[1] = oclTriCub33alpha[7]*A1 + oclTriCub33beta[7]*B1 + oclTriCub33gamma[7]*C1;
    cub33Q[2] = oclTriCub33alpha[7]*A2 + oclTriCub33beta[7]*B2 + oclTriCub33gamma[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[27]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [28] beta_7 A, alpha_7 B, gamma_7 C

    cub33Q[0] = oclTriCub33beta[7]*A0 + oclTriCub33alpha[7]*B0 + oclTriCub33gamma[7]*C0;
    cub33Q[1] = oclTriCub33beta[7]*A1 + oclTriCub33alpha[7]*B1 + oclTriCub33gamma[7]*C1;
    cub33Q[2] = oclTriCub33beta[7]*A2 + oclTriCub33alpha[7]*B2 + oclTriCub33gamma[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[28]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [29] gamma_7 A, beta_7 B, alpha_7 C

    cub33Q[0] = oclTriCub33gamma[7]*A0 + oclTriCub33beta[7]*B0 + oclTriCub33alpha[7]*C0;
    cub33Q[1] = oclTriCub33gamma[7]*A1 + oclTriCub33beta[7]*B1 + oclTriCub33alpha[7]*C1;
    cub33Q[2] = oclTriCub33gamma[7]*A2 + oclTriCub33beta[7]*B2 + oclTriCub33alpha[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[29]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [30] alpha_7 A, gamma_7 B, beta_7 C

    cub33Q[0] = oclTriCub33alpha[7]*A0 + oclTriCub33gamma[7]*B0 + oclTriCub33beta[7]*C0;
    cub33Q[1] = oclTriCub33alpha[7]*A1 + oclTriCub33gamma[7]*B1 + oclTriCub33beta[7]*C1;
    cub33Q[2] = oclTriCub33alpha[7]*A2 + oclTriCub33gamma[7]*B2 + oclTriCub33beta[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[30]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [31] gamma_7 A, alpha_7 B, beta_7 C

    cub33Q[0] = oclTriCub33gamma[7]*A0 + oclTriCub33alpha[7]*B0 + oclTriCub33beta[7]*C0;
    cub33Q[1] = oclTriCub33gamma[7]*A1 + oclTriCub33alpha[7]*B1 + oclTriCub33beta[7]*C1;
    cub33Q[2] = oclTriCub33gamma[7]*A2 + oclTriCub33alpha[7]*B2 + oclTriCub33beta[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[31]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // [32] beta_7 A, gamma_7 B, alpha_7 C

    cub33Q[0] = oclTriCub33beta[7]*A0 + oclTriCub33gamma[7]*B0 + oclTriCub33alpha[7]*C0;
    cub33Q[1] = oclTriCub33beta[7]*A1 + oclTriCub33gamma[7]*B1 + oclTriCub33alpha[7]*C1;
    cub33Q[2] = oclTriCub33beta[7]*A2 + oclTriCub33gamma[7]*B2 + oclTriCub33alpha[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[32]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    return (CL_TYPE4)( prefacStatic*finalSum0, prefacStatic*finalSum1, prefacStatic*finalSum2, 0. );
}

//______________________________________________________________________________

CL_TYPE4 ET_EFieldAndPotential_Cub33P( const CL_TYPE* P, __global const CL_TYPE* data )
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

    // [0] alpha_0 A, beta_0 B, gamma_0 C

    CL_TYPE cub33Q[3] = {
    		oclTriCub33alpha[0]*A0 + oclTriCub33beta[0]*B0 + oclTriCub33gamma[0]*C0,
			oclTriCub33alpha[0]*A1 + oclTriCub33beta[0]*B1 + oclTriCub33gamma[0]*C1,
			oclTriCub33alpha[0]*A2 + oclTriCub33beta[0]*B2 + oclTriCub33gamma[0]*C2 };

  	CL_TYPE oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	CL_TYPE prefacDynamic = oclTriCub33w[0]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[0]*oneOverAbsoluteValue);

    // [1] beta_0 A, alpha_0 B, gamma_0 C

    cub33Q[0] = oclTriCub33beta[0]*A0 + oclTriCub33alpha[0]*B0 + oclTriCub33gamma[0]*C0;
    cub33Q[1] = oclTriCub33beta[0]*A1 + oclTriCub33alpha[0]*B1 + oclTriCub33gamma[0]*C1;
    cub33Q[2] = oclTriCub33beta[0]*A2 + oclTriCub33alpha[0]*B2 + oclTriCub33gamma[0]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[1]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[1]*oneOverAbsoluteValue);

    // [2] gamma_0 A, beta_0 B, alpha_0 C

    cub33Q[0] = oclTriCub33gamma[0]*A0 + oclTriCub33beta[0]*B0 + oclTriCub33alpha[0]*C0;
    cub33Q[1] = oclTriCub33gamma[0]*A1 + oclTriCub33beta[0]*B1 + oclTriCub33alpha[0]*C1;
    cub33Q[2] = oclTriCub33gamma[0]*A2 + oclTriCub33beta[0]*B2 + oclTriCub33alpha[0]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[2]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[2]*oneOverAbsoluteValue);

    // [3] alpha_1 A, beta_1 B, gamma_1 C

    cub33Q[0] = oclTriCub33alpha[1]*A0 + oclTriCub33beta[1]*B0 + oclTriCub33gamma[1]*C0;
    cub33Q[1] = oclTriCub33alpha[1]*A1 + oclTriCub33beta[1]*B1 + oclTriCub33gamma[1]*C1;
    cub33Q[2] = oclTriCub33alpha[1]*A2 + oclTriCub33beta[1]*B2 + oclTriCub33gamma[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[3]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[3]*oneOverAbsoluteValue);

    // [4] beta_1 A, alpha_1 B, gamma_1 C

    cub33Q[0] = oclTriCub33beta[1]*A0 + oclTriCub33alpha[1]*B0 + oclTriCub33gamma[1]*C0;
    cub33Q[1] = oclTriCub33beta[1]*A1 + oclTriCub33alpha[1]*B1 + oclTriCub33gamma[1]*C1;
    cub33Q[2] = oclTriCub33beta[1]*A2 + oclTriCub33alpha[1]*B2 + oclTriCub33gamma[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[4]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[4]*oneOverAbsoluteValue);

    // [5] gamma_1 A, beta_1 B, alpha_1 C

    cub33Q[0] = oclTriCub33gamma[1]*A0 + oclTriCub33beta[1]*B0 + oclTriCub33alpha[1]*C0;
    cub33Q[1] = oclTriCub33gamma[1]*A1 + oclTriCub33beta[1]*B1 + oclTriCub33alpha[1]*C1;
    cub33Q[2] = oclTriCub33gamma[1]*A2 + oclTriCub33beta[1]*B2 + oclTriCub33alpha[1]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[5]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[5]*oneOverAbsoluteValue);

    // [6] alpha_2 A, beta_2 B, gamma_2 C

    cub33Q[0] = oclTriCub33alpha[2]*A0 + oclTriCub33beta[2]*B0 + oclTriCub33gamma[2]*C0;
    cub33Q[1] = oclTriCub33alpha[2]*A1 + oclTriCub33beta[2]*B1 + oclTriCub33gamma[2]*C1;
    cub33Q[2] = oclTriCub33alpha[2]*A2 + oclTriCub33beta[2]*B2 + oclTriCub33gamma[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[6]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[6]*oneOverAbsoluteValue);

    // [7] beta_2 A, alpha_2 B, gamma_2 C

    cub33Q[0] = oclTriCub33beta[2]*A0 + oclTriCub33alpha[2]*B0 + oclTriCub33gamma[2]*C0;
    cub33Q[1] = oclTriCub33beta[2]*A1 + oclTriCub33alpha[2]*B1 + oclTriCub33gamma[2]*C1;
    cub33Q[2] = oclTriCub33beta[2]*A2 + oclTriCub33alpha[2]*B2 + oclTriCub33gamma[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[7]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[7]*oneOverAbsoluteValue);

    // [8] gamma_2 A, beta_2 B, alpha_2 C

    cub33Q[0] = oclTriCub33gamma[2]*A0 + oclTriCub33beta[2]*B0 + oclTriCub33alpha[2]*C0;
    cub33Q[1] = oclTriCub33gamma[2]*A1 + oclTriCub33beta[2]*B1 + oclTriCub33alpha[2]*C1;
    cub33Q[2] = oclTriCub33gamma[2]*A2 + oclTriCub33beta[2]*B2 + oclTriCub33alpha[2]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[8]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[8]*oneOverAbsoluteValue);

    // [9] alpha_3 A, beta_3 B, gamma_3 C

    cub33Q[0] = oclTriCub33alpha[3]*A0 + oclTriCub33beta[3]*B0 + oclTriCub33gamma[3]*C0;
    cub33Q[1] = oclTriCub33alpha[3]*A1 + oclTriCub33beta[3]*B1 + oclTriCub33gamma[3]*C1;
    cub33Q[2] = oclTriCub33alpha[3]*A2 + oclTriCub33beta[3]*B2 + oclTriCub33gamma[3]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[9]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[9]*oneOverAbsoluteValue);

    // [10] beta_3 A, alpha_3 B, gamma_3 C

    cub33Q[0] = oclTriCub33beta[3]*A0 + oclTriCub33alpha[3]*B0 + oclTriCub33gamma[3]*C0;
    cub33Q[1] = oclTriCub33beta[3]*A1 + oclTriCub33alpha[3]*B1 + oclTriCub33gamma[3]*C1;
    cub33Q[2] = oclTriCub33beta[3]*A2 + oclTriCub33alpha[3]*B2 + oclTriCub33gamma[3]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[10]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[10]*oneOverAbsoluteValue);

    // [11] gamma_3 A, beta_3 B, alpha_3 C

    cub33Q[0] = oclTriCub33gamma[3]*A0 + oclTriCub33beta[3]*B0 + oclTriCub33alpha[3]*C0;
    cub33Q[1] = oclTriCub33gamma[3]*A1 + oclTriCub33beta[3]*B1 + oclTriCub33alpha[3]*C1;
    cub33Q[2] = oclTriCub33gamma[3]*A2 + oclTriCub33beta[3]*B2 + oclTriCub33alpha[3]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[11]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[11]*oneOverAbsoluteValue);

    // [12] alpha_4 A, beta_4 B, gamma_4 C

    cub33Q[0] = oclTriCub33alpha[4]*A0 + oclTriCub33beta[4]*B0 + oclTriCub33gamma[4]*C0;
    cub33Q[1] = oclTriCub33alpha[4]*A1 + oclTriCub33beta[4]*B1 + oclTriCub33gamma[4]*C1;
    cub33Q[2] = oclTriCub33alpha[4]*A2 + oclTriCub33beta[4]*B2 + oclTriCub33gamma[4]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[12]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[12]*oneOverAbsoluteValue);

    // [13] beta_4 A, alpha_4 B, gamma_4 C

    cub33Q[0] = oclTriCub33beta[4]*A0 + oclTriCub33alpha[4]*B0 + oclTriCub33gamma[4]*C0;
    cub33Q[1] = oclTriCub33beta[4]*A1 + oclTriCub33alpha[4]*B1 + oclTriCub33gamma[4]*C1;
    cub33Q[2] = oclTriCub33beta[4]*A2 + oclTriCub33alpha[4]*B2 + oclTriCub33gamma[4]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[13]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[13]*oneOverAbsoluteValue);

    // [14] gamma_4 A, beta_4 B, alpha_4 C

    cub33Q[0] = oclTriCub33gamma[4]*A0 + oclTriCub33beta[4]*B0 + oclTriCub33alpha[4]*C0;
    cub33Q[1] = oclTriCub33gamma[4]*A1 + oclTriCub33beta[4]*B1 + oclTriCub33alpha[4]*C1;
    cub33Q[2] = oclTriCub33gamma[4]*A2 + oclTriCub33beta[4]*B2 + oclTriCub33alpha[4]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[14]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[14]*oneOverAbsoluteValue);

    // [15] alpha_5 A, beta_5 B, gamma_5 C

    cub33Q[0] = oclTriCub33alpha[5]*A0 + oclTriCub33beta[5]*B0 + oclTriCub33gamma[5]*C0;
    cub33Q[1] = oclTriCub33alpha[5]*A1 + oclTriCub33beta[5]*B1 + oclTriCub33gamma[5]*C1;
    cub33Q[2] = oclTriCub33alpha[5]*A2 + oclTriCub33beta[5]*B2 + oclTriCub33gamma[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[15]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[15]*oneOverAbsoluteValue);

    // [16] beta_5 A, alpha_5 B, gamma_5 C

    cub33Q[0] = oclTriCub33beta[5]*A0 + oclTriCub33alpha[5]*B0 + oclTriCub33gamma[5]*C0;
    cub33Q[1] = oclTriCub33beta[5]*A1 + oclTriCub33alpha[5]*B1 + oclTriCub33gamma[5]*C1;
    cub33Q[2] = oclTriCub33beta[5]*A2 + oclTriCub33alpha[5]*B2 + oclTriCub33gamma[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[16]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[16]*oneOverAbsoluteValue);

    // [17] gamma_5 A, beta_5 B, alpha_5 C

    cub33Q[0] = oclTriCub33gamma[5]*A0 + oclTriCub33beta[5]*B0 + oclTriCub33alpha[5]*C0;
    cub33Q[1] = oclTriCub33gamma[5]*A1 + oclTriCub33beta[5]*B1 + oclTriCub33alpha[5]*C1;
    cub33Q[2] = oclTriCub33gamma[5]*A2 + oclTriCub33beta[5]*B2 + oclTriCub33alpha[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[17]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[17]*oneOverAbsoluteValue);

    // [18] alpha_5 A, gamma_5 B, beta_5 C

    cub33Q[0] = oclTriCub33alpha[5]*A0 + oclTriCub33gamma[5]*B0 + oclTriCub33beta[5]*C0;
    cub33Q[1] = oclTriCub33alpha[5]*A1 + oclTriCub33gamma[5]*B1 + oclTriCub33beta[5]*C1;
    cub33Q[2] = oclTriCub33alpha[5]*A2 + oclTriCub33gamma[5]*B2 + oclTriCub33beta[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[18]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[18]*oneOverAbsoluteValue);

    // [19] gamma_5 A, alpha_5 B, beta_5 C

    cub33Q[0] = oclTriCub33gamma[5]*A0 + oclTriCub33alpha[5]*B0 + oclTriCub33beta[5]*C0;
    cub33Q[1] = oclTriCub33gamma[5]*A1 + oclTriCub33alpha[5]*B1 + oclTriCub33beta[5]*C1;
    cub33Q[2] = oclTriCub33gamma[5]*A2 + oclTriCub33alpha[5]*B2 + oclTriCub33beta[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[19]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[19]*oneOverAbsoluteValue);

    // [20] beta_5 A, gamma_5 B, alpha_5 C

    cub33Q[0] = oclTriCub33beta[5]*A0 + oclTriCub33gamma[5]*B0 + oclTriCub33alpha[5]*C0;
    cub33Q[1] = oclTriCub33beta[5]*A1 + oclTriCub33gamma[5]*B1 + oclTriCub33alpha[5]*C1;
    cub33Q[2] = oclTriCub33beta[5]*A2 + oclTriCub33gamma[5]*B2 + oclTriCub33alpha[5]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[20]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[20]*oneOverAbsoluteValue);

    // [21] alpha_6 A, beta_6 B, gamma_6 C

    cub33Q[0] = oclTriCub33alpha[6]*A0 + oclTriCub33beta[6]*B0 + oclTriCub33gamma[6]*C0;
    cub33Q[1] = oclTriCub33alpha[6]*A1 + oclTriCub33beta[6]*B1 + oclTriCub33gamma[6]*C1;
    cub33Q[2] = oclTriCub33alpha[6]*A2 + oclTriCub33beta[6]*B2 + oclTriCub33gamma[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[21]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[21]*oneOverAbsoluteValue);

    // [22] beta_6 A, alpha_6 B, gamma_6 C

    cub33Q[0] = oclTriCub33beta[6]*A0 + oclTriCub33alpha[6]*B0 + oclTriCub33gamma[6]*C0;
    cub33Q[1] = oclTriCub33beta[6]*A1 + oclTriCub33alpha[6]*B1 + oclTriCub33gamma[6]*C1;
    cub33Q[2] = oclTriCub33beta[6]*A2 + oclTriCub33alpha[6]*B2 + oclTriCub33gamma[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[22]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[22]*oneOverAbsoluteValue);

    // [23] gamma_6 A, beta_6 B, alpha_6 C

    cub33Q[0] = oclTriCub33gamma[6]*A0 + oclTriCub33beta[6]*B0 + oclTriCub33alpha[6]*C0;
    cub33Q[1] = oclTriCub33gamma[6]*A1 + oclTriCub33beta[6]*B1 + oclTriCub33alpha[6]*C1;
    cub33Q[2] = oclTriCub33gamma[6]*A2 + oclTriCub33beta[6]*B2 + oclTriCub33alpha[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[23]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[23]*oneOverAbsoluteValue);

    // [24] alpha_6 A, gamma_6 B, beta_6 C

    cub33Q[0] = oclTriCub33alpha[6]*A0 + oclTriCub33gamma[6]*B0 + oclTriCub33beta[6]*C0;
    cub33Q[1] = oclTriCub33alpha[6]*A1 + oclTriCub33gamma[6]*B1 + oclTriCub33beta[6]*C1;
    cub33Q[2] = oclTriCub33alpha[6]*A2 + oclTriCub33gamma[6]*B2 + oclTriCub33beta[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[24]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[24]*oneOverAbsoluteValue);

    // [25] gamma_6 A, alpha_6 B, beta_6 C

    cub33Q[0] = oclTriCub33gamma[6]*A0 + oclTriCub33alpha[6]*B0 + oclTriCub33beta[6]*C0;
    cub33Q[1] = oclTriCub33gamma[6]*A1 + oclTriCub33alpha[6]*B1 + oclTriCub33beta[6]*C1;
    cub33Q[2] = oclTriCub33gamma[6]*A2 + oclTriCub33alpha[6]*B2 + oclTriCub33beta[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[25]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[25]*oneOverAbsoluteValue);

    // [26] beta_6 A, gamma_6 B, alpha_6 C

    cub33Q[0] = oclTriCub33beta[6]*A0 + oclTriCub33gamma[6]*B0 + oclTriCub33alpha[6]*C0;
    cub33Q[1] = oclTriCub33beta[6]*A1 + oclTriCub33gamma[6]*B1 + oclTriCub33alpha[6]*C1;
    cub33Q[2] = oclTriCub33beta[6]*A2 + oclTriCub33gamma[6]*B2 + oclTriCub33alpha[6]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[26]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[26]*oneOverAbsoluteValue);

    // [27] alpha_7 A, beta_7 B, gamma_7 C

    cub33Q[0] = oclTriCub33alpha[7]*A0 + oclTriCub33beta[7]*B0 + oclTriCub33gamma[7]*C0;
    cub33Q[1] = oclTriCub33alpha[7]*A1 + oclTriCub33beta[7]*B1 + oclTriCub33gamma[7]*C1;
    cub33Q[2] = oclTriCub33alpha[7]*A2 + oclTriCub33beta[7]*B2 + oclTriCub33gamma[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[27]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[27]*oneOverAbsoluteValue);

    // [28] beta_7 A, alpha_7 B, gamma_7 C

    cub33Q[0] = oclTriCub33beta[7]*A0 + oclTriCub33alpha[7]*B0 + oclTriCub33gamma[7]*C0;
    cub33Q[1] = oclTriCub33beta[7]*A1 + oclTriCub33alpha[7]*B1 + oclTriCub33gamma[7]*C1;
    cub33Q[2] = oclTriCub33beta[7]*A2 + oclTriCub33alpha[7]*B2 + oclTriCub33gamma[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[28]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[28]*oneOverAbsoluteValue);

    // [29] gamma_7 A, beta_7 B, alpha_7 C

    cub33Q[0] = oclTriCub33gamma[7]*A0 + oclTriCub33beta[7]*B0 + oclTriCub33alpha[7]*C0;
    cub33Q[1] = oclTriCub33gamma[7]*A1 + oclTriCub33beta[7]*B1 + oclTriCub33alpha[7]*C1;
    cub33Q[2] = oclTriCub33gamma[7]*A2 + oclTriCub33beta[7]*B2 + oclTriCub33alpha[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[29]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[29]*oneOverAbsoluteValue);

    // [30] alpha_7 A, gamma_7 B, beta_7 C

    cub33Q[0] = oclTriCub33alpha[7]*A0 + oclTriCub33gamma[7]*B0 + oclTriCub33beta[7]*C0;
    cub33Q[1] = oclTriCub33alpha[7]*A1 + oclTriCub33gamma[7]*B1 + oclTriCub33beta[7]*C1;
    cub33Q[2] = oclTriCub33alpha[7]*A2 + oclTriCub33gamma[7]*B2 + oclTriCub33beta[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[30]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[30]*oneOverAbsoluteValue);

    // [31] gamma_7 A, alpha_7 B, beta_7 C

    cub33Q[0] = oclTriCub33gamma[7]*A0 + oclTriCub33alpha[7]*B0 + oclTriCub33beta[7]*C0;
    cub33Q[1] = oclTriCub33gamma[7]*A1 + oclTriCub33alpha[7]*B1 + oclTriCub33beta[7]*C1;
    cub33Q[2] = oclTriCub33gamma[7]*A2 + oclTriCub33alpha[7]*B2 + oclTriCub33beta[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[31]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[31]*oneOverAbsoluteValue);

    // [32] beta_7 A, gamma_7 B, alpha_7 C

    cub33Q[0] = oclTriCub33beta[7]*A0 + oclTriCub33gamma[7]*B0 + oclTriCub33alpha[7]*C0;
    cub33Q[1] = oclTriCub33beta[7]*A1 + oclTriCub33gamma[7]*B1 + oclTriCub33alpha[7]*C1;
    cub33Q[2] = oclTriCub33beta[7]*A2 + oclTriCub33gamma[7]*B2 + oclTriCub33alpha[7]*C2;

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub33Q,P);
  	prefacDynamic = oclTriCub33w[32]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (oclTriCub33w[32]*oneOverAbsoluteValue);

    return (CL_TYPE4)( prefacStatic*finalSum0, prefacStatic*finalSum1, prefacStatic*finalSum2, prefacStatic*finalSum3 );
}

#endif /* KEMFIELD_ELECTROSTATICCUBATURETRIANGLE_33POINT_CL */
