#ifndef KEMFIELD_ELECTROSTATICCUBATURERECTANGLE_12POINT_CUH
#define KEMFIELD_ELECTROSTATICCUBATURERECTANGLE_12POINT_CUH

// CUDA kernel for rectangle boundary integrator with 12-point Gaussian cubature
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
// * ER_Potential_Cub12P: 896
// * ER_EField_Cub12P: 768
// * ER_EFieldAndPotential_Cub12P: 768

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

__constant__ CU_TYPE cuRectCub12w[12];

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE ER_Potential_Cub12P( const CU_TYPE* P, const CU_TYPE* data )
{
    const CU_TYPE prefacStatic = data[0] * data[1] * M_ONEOVER_4PI_EPS0;
    CU_TYPE finalSum = 0.;

	const CU_TYPE a2 = 0.5*data[0];
	const CU_TYPE b2 = 0.5*data[1];

	const CU_TYPE cen0 = data[2] + (a2*data[5]) + (b2*data[8]);
	const CU_TYPE cen1 = data[3] + (a2*data[6]) + (b2*data[9]);
	const CU_TYPE cen2 = data[4] + (a2*data[7]) + (b2*data[10]);

    const CU_TYPE r = SQRT( 6./7. );
    const CU_TYPE s = SQRT( (114.-3.*SQRT(583.)) / 287. );
    const CU_TYPE t = SQRT( (114.+3.*SQRT(583.)) / 287. );

    // 0

    CU_TYPE x = r;
    CU_TYPE y = 0.;

    CU_TYPE cub12Q[3] = {
    		cen0 + (x*a2)*data[5] + (y*b2)*data[8],
			cen1 + (x*a2)*data[6] + (y*b2)*data[9],
			cen2 + (x*a2)*data[7] + (y*b2)*data[10]
    };

   	finalSum += (cuRectCub12w[0]*OneOverR(cub12Q,P));

   	// 1

    x = -r;
    y = 0.;

   	cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
   	cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
   	cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

   	finalSum += (cuRectCub12w[1]*OneOverR(cub12Q,P));

   	// 2

    x = 0.;
    y = r;

   	cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
   	cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
   	cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

   	finalSum += (cuRectCub12w[2]*OneOverR(cub12Q,P));

   	// 3

    x = 0.;
    y = -r;

   	cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
   	cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
   	cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

   	finalSum += (cuRectCub12w[3]*OneOverR(cub12Q,P));

   	// 4

    x = s;
    y = s;

   	cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
   	cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
   	cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

   	finalSum += (cuRectCub12w[4]*OneOverR(cub12Q,P));

   	// 5

    x = s;
    y = -s;

   	cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
   	cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
   	cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

   	finalSum += (cuRectCub12w[5]*OneOverR(cub12Q,P));

   	// 6

    x = -s;
    y = s;

   	cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
   	cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
   	cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

   	finalSum += (cuRectCub12w[6]*OneOverR(cub12Q,P));

    // 7

    x = -s;
    y = -s;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub12w[7]*OneOverR(cub12Q,P));

    // 8

    x = t;
    y = t;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub12w[8]*OneOverR(cub12Q,P));

    // 9

    x = t;
    y = -t;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub12w[9]*OneOverR(cub12Q,P));

    // 10

    x = -t;
    y = t;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub12w[10]*OneOverR(cub12Q,P));

    // 11

    x = -t;
    y = -t;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    finalSum += (cuRectCub12w[11]*OneOverR(cub12Q,P));

    return (prefacStatic*finalSum);
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE4 ER_EField_Cub12P(const CU_TYPE* P, const CU_TYPE* data)
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

    const CU_TYPE r = SQRT( 6./7. );
    const CU_TYPE s = SQRT( (114.-3.*SQRT(583.)) / 287. );
    const CU_TYPE t = SQRT( (114.+3.*SQRT(583.)) / 287. );

    // 0

    CU_TYPE x = r;
    CU_TYPE y = 0.;

    CU_TYPE cub12Q[3] = {
            cen0 + (x*a2)*data[5] + (y*b2)*data[8],
            cen1 + (x*a2)*data[6] + (y*b2)*data[9],
            cen2 + (x*a2)*data[7] + (y*b2)*data[10]
    };

  	CU_TYPE oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
  	CU_TYPE prefacDynamic = cuRectCub12w[0]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 1

    x = -r;
    y = 0.;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

  	oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
  	prefacDynamic = cuRectCub12w[1]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 2

    x = 0.;
    y = r;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

  	oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
  	prefacDynamic = cuRectCub12w[2]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 3

    x = 0.;
    y = -r;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

  	oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
  	prefacDynamic = cuRectCub12w[3]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 4

    x = s;
    y = s;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

  	oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
  	prefacDynamic = cuRectCub12w[4]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 5

    x = s;
    y = -s;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

  	oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
  	prefacDynamic = cuRectCub12w[5]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 6

    x = -s;
    y = s;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

  	oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
  	prefacDynamic = cuRectCub12w[6]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 7

    x = -s;
    y = -s;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
    prefacDynamic = cuRectCub12w[7]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 8

    x = t;
    y = t;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
    prefacDynamic = cuRectCub12w[8]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 9

    x = t;
    y = -t;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
    prefacDynamic = cuRectCub12w[9]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 10

    x = -t;
    y = t;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
    prefacDynamic = cuRectCub12w[10]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    // 11

    x = -t;
    y = -t;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
    prefacDynamic = cuRectCub12w[11]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);

    return MAKECU4( prefacStatic*finalSum0, prefacStatic*finalSum1, prefacStatic*finalSum2, 0. );
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE4 ER_EFieldAndPotential_Cub12P(const CU_TYPE* P, const CU_TYPE* data)
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

    const CU_TYPE r = SQRT( 6./7. );
    const CU_TYPE s = SQRT( (114.-3.*SQRT(583.)) / 287. );
    const CU_TYPE t = SQRT( (114.+3.*SQRT(583.)) / 287. );

    // 0

    CU_TYPE x = r;
    CU_TYPE y = 0.;

    CU_TYPE cub12Q[3] = {
            cen0 + (x*a2)*data[5] + (y*b2)*data[8],
            cen1 + (x*a2)*data[6] + (y*b2)*data[9],
            cen2 + (x*a2)*data[7] + (y*b2)*data[10]
    };

  	CU_TYPE oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
  	CU_TYPE prefacDynamic = cuRectCub12w[0]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
   	finalSum3 += (cuRectCub12w[0]*OneOverR(cub12Q,P));

    // 1

    x = -r;
    y = 0.;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

  	oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
  	prefacDynamic = cuRectCub12w[1]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
   	finalSum3 += (cuRectCub12w[1]*OneOverR(cub12Q,P));

    // 2

    x = 0.;
    y = r;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

  	oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
  	prefacDynamic = cuRectCub12w[2]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
   	finalSum3 += (cuRectCub12w[2]*OneOverR(cub12Q,P));

    // 3

    x = 0.;
    y = -r;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

  	oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
  	prefacDynamic = cuRectCub12w[3]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
   	finalSum3 += (cuRectCub12w[3]*OneOverR(cub12Q,P));

    // 4

    x = s;
    y = s;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

  	oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
  	prefacDynamic = cuRectCub12w[4]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
   	finalSum3 += (cuRectCub12w[4]*OneOverR(cub12Q,P));

    // 5

    x = s;
    y = -s;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

  	oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
  	prefacDynamic = cuRectCub12w[5]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
   	finalSum3 += (cuRectCub12w[5]*OneOverR(cub12Q,P));

    // 6

    x = -s;
    y = s;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

  	oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
  	prefacDynamic = cuRectCub12w[6]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
   	finalSum3 += (cuRectCub12w[6]*OneOverR(cub12Q,P));

    // 7

    x = -s;
    y = -s;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
    prefacDynamic = cuRectCub12w[7]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub12w[7]*OneOverR(cub12Q,P));

    // 8

    x = t;
    y = t;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
    prefacDynamic = cuRectCub12w[8]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub12w[8]*OneOverR(cub12Q,P));

    // 9

    x = t;
    y = -t;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
    prefacDynamic = cuRectCub12w[9]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub12w[9]*OneOverR(cub12Q,P));

    // 10

    x = -t;
    y = t;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
    prefacDynamic = cuRectCub12w[10]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub12w[10]*OneOverR(cub12Q,P));

    // 11

    x = -t;
    y = -t;

    cub12Q[0] = cen0 + (x*a2)*data[5] + (y*b2)*data[8];
    cub12Q[1] = cen1 + (x*a2)*data[6] + (y*b2)*data[9];
    cub12Q[2] = cen2 + (x*a2)*data[7] + (y*b2)*data[10];

    oneOverAbsoluteValue = OneOverR_VecToArr(distVector,cub12Q,P);
    prefacDynamic = cuRectCub12w[11]*POW3(oneOverAbsoluteValue);

    finalSum0 += (prefacDynamic*distVector[0]);
    finalSum1 += (prefacDynamic*distVector[1]);
    finalSum2 += (prefacDynamic*distVector[2]);
    finalSum3 += (cuRectCub12w[11]*OneOverR(cub12Q,P));

    return MAKECU4( prefacStatic*finalSum0, prefacStatic*finalSum1, prefacStatic*finalSum2, prefacStatic*finalSum3 );
}

#endif /* KEMFIELD_ELECTROSTATICCUBATURERECTANGLE_12POINT_CUH */
