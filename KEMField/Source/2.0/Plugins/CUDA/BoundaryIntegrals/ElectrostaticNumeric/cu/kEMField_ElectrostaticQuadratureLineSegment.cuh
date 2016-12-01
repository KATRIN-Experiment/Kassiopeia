#ifndef KEMFIELD_ELECTROSTATICQUADRATURELINESEGMENT_CUH
#define KEMFIELD_ELECTROSTATICQUADRATURELINESEGMENT_CUH

// CUDA kernel for line segment boundary integrator with 4-node and 16-node Gaussian quadrature
// Detailed information on the quadrature implementation can be found in the CPU code,
// class 'KElectrostaticQuadratureLineSegmentIntegrator'.
// Author: Daniel Hilk
//
// This kernel version is optimized regarding thread block size and speed for compute
// devices providing only scalar units, but runs as well very efficient on devices
// providing vector units additionally. The optimal sizes for the used hardware
// will be set automatically.
//
// Recommended thread block sizes by CUDA Occupancy API for NVIDIA Tesla K40c:
// * EL_Potential_Quad4N: 384
// * EL_EField_Quad4N: 512
// * EL_EFieldAndPotential_Quad4N: 512
// * EL_Potential_Quad16N: 384
// * EL_EField_Quad16N: 512
// * EL_EFieldAndPotential_Quad16N: 512

#include "kEMField_LineSegment.cuh"

// Wire geometry definition (as defined by the streamers in KLineSegment.hh):
//
// data[0..2]: P0[0..2]
// data[3..5]: P1[0..2]
// data[6]:    diameter

//______________________________________________________________________________

	// Gaussian nodes and weights

	__constant__ CU_TYPE cuLineQuadx4[2];
	__constant__ CU_TYPE cuLineQuadw4[2];

	__constant__ CU_TYPE cuLineQuadx16[8];
	__constant__ CU_TYPE cuLineQuadw16[8];

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE EL_Potential_Quad4N( const CU_TYPE* P, const CU_TYPE* data )
{
	const CU_TYPE lineCenter0 = (data[0] + data[3])*.5;
	const CU_TYPE lineCenter1 = (data[1] + data[4])*.5;
	const CU_TYPE lineCenter2 = (data[2] + data[5])*.5;

	CU_TYPE lineLength[1];
	Line_Length( lineLength, data );
	const CU_TYPE halfLength = 0.5 * lineLength[0]; // only half of length is needed in iteration

    CU_TYPE prefacUnit = 1./lineLength[0];
    CU_TYPE lineUnit0 = prefacUnit * (data[3] - data[0]);
    CU_TYPE lineUnit1 = prefacUnit * (data[4] - data[1]);
    CU_TYPE lineUnit2 = prefacUnit * (data[5] - data[2]);

    CU_TYPE sum = 0.;

    CU_TYPE posDir0 = 0.;
    CU_TYPE posDir1 = 0.;
    CU_TYPE posDir2 = 0.;

    CU_TYPE negDir0 = 0.;
    CU_TYPE negDir1 = 0.;
    CU_TYPE negDir2 = 0.;

    CU_TYPE posMag = 0.;
    CU_TYPE negMag = 0.;

	CU_TYPE weightFac = 0.;
	CU_TYPE nodeIt = 0.;

	for( unsigned short i=0; i<2; i++ ) {
		weightFac = cuLineQuadw4[i] * halfLength;
		nodeIt = cuLineQuadx4[i] * halfLength;

		// reset variables for magnitude
		posMag = 0.;
		negMag = 0.;

		posDir0 = P[0] - ( lineCenter0 + (lineUnit0*nodeIt) );
		posMag += POW2(posDir0);
		negDir0 = P[0] - ( lineCenter0 - (lineUnit0*nodeIt) );
		negMag += POW2(negDir0);

		posDir1 = P[1] - ( lineCenter1 + (lineUnit1*nodeIt) );
		posMag += POW2(posDir1);
		negDir1 = P[1] - ( lineCenter1 - (lineUnit1*nodeIt));
		negMag += POW2(negDir1);

		posDir2 = P[2] - ( lineCenter2 + (lineUnit2*nodeIt) );
		posMag += POW2(posDir2);
		negDir2 = P[2] - ( lineCenter2 - (lineUnit2*nodeIt) );
		negMag += POW2(negDir2);

		posMag = 1. / SQRT( posMag );
		negMag = 1. / SQRT( negMag );

		sum += weightFac * (posMag + negMag);
	}

	const CU_TYPE oneOverEps0 = 1./(M_EPS0);

    return (sum*data[6]*0.25*oneOverEps0);
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE4 EL_EField_Quad4N(const CU_TYPE* P, const CU_TYPE* data)
{
	const CU_TYPE lineCenter0 = (data[0] + data[3])*.5;
	const CU_TYPE lineCenter1 = (data[1] + data[4])*.5;
	const CU_TYPE lineCenter2 = (data[2] + data[5])*.5;

	CU_TYPE lineLength[1];
	Line_Length( lineLength, data );
	CU_TYPE halfLength = 0.5 * lineLength[0]; // only half of length is needed in iteration

    CU_TYPE prefacUnit = 1./lineLength[0];
    CU_TYPE lineUnit0 = prefacUnit * (data[3] - data[0]);
    CU_TYPE lineUnit1 = prefacUnit * (data[4] - data[1]);
    CU_TYPE lineUnit2 = prefacUnit * (data[5] - data[2]);

    CU_TYPE sum0 = 0.;
    CU_TYPE sum1 = 0.;
    CU_TYPE sum2 = 0.;

    CU_TYPE posDir0 = 0.;
    CU_TYPE posDir1 = 0.;
    CU_TYPE posDir2 = 0.;

    CU_TYPE negDir0 = 0.;
    CU_TYPE negDir1 = 0.;
    CU_TYPE negDir2 = 0.;

    CU_TYPE posMag = 0.;
    CU_TYPE negMag = 0.;

	CU_TYPE weightFac = 0.;
	CU_TYPE nodeIt = 0.;

	for( unsigned short i=0; i<2; i++ ) {
		weightFac = cuLineQuadw4[i]*halfLength;
		nodeIt = cuLineQuadx4[i]*halfLength;

		// reset variables for magnitude
		posMag = 0.;
		negMag = 0.;

		posDir0 = P[0] - ( lineCenter0 + (lineUnit0*nodeIt) );
		posDir1 = P[1] - ( lineCenter1 + (lineUnit1*nodeIt) );
		posDir2 = P[2] - ( lineCenter2 + (lineUnit2*nodeIt) );

		negDir0 = P[0] - ( lineCenter0 - (lineUnit0*nodeIt) );
		negDir1 = P[1] - ( lineCenter1 - (lineUnit1*nodeIt) );
		negDir2 = P[2] - ( lineCenter2 - (lineUnit2*nodeIt) );

		posMag = POW2(posDir0) + POW2(posDir1) + POW2(posDir2);
		posMag = 1. / SQRT( posMag );
		posMag = POW3( posMag );

		negMag = POW2(negDir0) + POW2(negDir1) + POW2(negDir2);
		negMag = 1. / SQRT( negMag );
		negMag = POW3( negMag );

		sum0 += weightFac * (posMag*posDir0 + negMag*negDir0);
		sum1 += weightFac * (posMag*posDir1 + negMag*negDir1);
		sum2 += weightFac * (posMag*posDir2 + negMag*negDir2);
	}

	const CU_TYPE prefac = (0.25*data[6])/(M_EPS0);

    return MAKECU4( prefac*sum0, prefac*sum1, prefac*sum2, 0. );
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE4 EL_EFieldAndPotential_Quad4N(const CU_TYPE* P, const CU_TYPE* data)
{
	const CU_TYPE lineCenter0 = (data[0] + data[3])*.5;
	const CU_TYPE lineCenter1 = (data[1] + data[4])*.5;
	const CU_TYPE lineCenter2 = (data[2] + data[5])*.5;

	CU_TYPE lineLength[1];
	Line_Length( lineLength, data );
	CU_TYPE halfLength = 0.5 * lineLength[0]; // only half of length is needed in iteration

    CU_TYPE prefacUnit = 1./lineLength[0];
    CU_TYPE lineUnit0 = prefacUnit * (data[3] - data[0]);
    CU_TYPE lineUnit1 = prefacUnit * (data[4] - data[1]);
    CU_TYPE lineUnit2 = prefacUnit * (data[5] - data[2]);

	CU_TYPE sum0 = 0.; /* field */
	CU_TYPE sum1 = 0.;
	CU_TYPE sum2 = 0.;
	CU_TYPE sum3 = 0.; /* potential */

    CU_TYPE posDir0 = 0.;
    CU_TYPE posDir1 = 0.;
    CU_TYPE posDir2 = 0.;

    CU_TYPE negDir0 = 0.;
    CU_TYPE negDir1 = 0.;
    CU_TYPE negDir2 = 0.;

    CU_TYPE posMag = 0.;
    CU_TYPE negMag = 0.;

	CU_TYPE weightFac = 0.;
	CU_TYPE nodeIt = 0.;

	for( unsigned short i=0; i<2; i++ ) {
		weightFac = cuLineQuadw4[i]*halfLength;
		nodeIt = cuLineQuadx4[i]*halfLength;

		// reset variables for magnitude
		posMag = 0.;
		negMag = 0.;

		posDir0 = P[0] - ( lineCenter0 + (lineUnit0*nodeIt) );
		posDir1 = P[1] - ( lineCenter1 + (lineUnit1*nodeIt) );
		posDir2 = P[2] - ( lineCenter2 + (lineUnit2*nodeIt) );

		negDir0 = P[0] - ( lineCenter0 - (lineUnit0*nodeIt) );
		negDir1 = P[1] - ( lineCenter1 - (lineUnit1*nodeIt) );
		negDir2 = P[2] - ( lineCenter2 - (lineUnit2*nodeIt) );

		posMag = POW2(posDir0) + POW2(posDir1) + POW2(posDir2);
		posMag = 1. / SQRT( posMag );

		negMag = POW2(negDir0) + POW2(negDir1) + POW2(negDir2);
		negMag = 1. / SQRT( negMag );

		sum3 += weightFac * (posMag + negMag);

		posMag = POW3( posMag );
		negMag = POW3( negMag );

		sum0 += weightFac * (posMag*posDir0 + negMag*negDir0);
		sum1 += weightFac * (posMag*posDir1 + negMag*negDir1);
		sum2 += weightFac * (posMag*posDir2 + negMag*negDir2);
	}

	const CU_TYPE prefac = (0.25*data[6])/(M_EPS0);

    return MAKECU4( prefac*sum0, prefac*sum1, prefac*sum2, prefac*sum3 );
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE EL_Potential_Quad16N( const CU_TYPE* P, const CU_TYPE* data )
{
	const CU_TYPE lineCenter0 = (data[0] + data[3])*.5;
	const CU_TYPE lineCenter1 = (data[1] + data[4])*.5;
	const CU_TYPE lineCenter2 = (data[2] + data[5])*.5;

	CU_TYPE lineLength[1];
	Line_Length( lineLength, data );
	const CU_TYPE halfLength = 0.5 * lineLength[0]; // only half of length is needed in iteration

    CU_TYPE prefacUnit = 1./lineLength[0];
    CU_TYPE lineUnit0 = prefacUnit * (data[3] - data[0]);
    CU_TYPE lineUnit1 = prefacUnit * (data[4] - data[1]);
    CU_TYPE lineUnit2 = prefacUnit * (data[5] - data[2]);

    CU_TYPE sum = 0.;

    CU_TYPE posDir0 = 0.;
    CU_TYPE posDir1 = 0.;
    CU_TYPE posDir2 = 0.;

    CU_TYPE negDir0 = 0.;
    CU_TYPE negDir1 = 0.;
    CU_TYPE negDir2 = 0.;

    CU_TYPE posMag = 0.;
    CU_TYPE negMag = 0.;

	CU_TYPE weightFac = 0.;
	CU_TYPE nodeIt = 0.;

	for( unsigned short i=0; i<8; i++ ) {
		weightFac = cuLineQuadw16[i] * halfLength;
		nodeIt = cuLineQuadx16[i] * halfLength;

		// reset variables for magnitude
		posMag = 0.;
		negMag = 0.;

		posDir0 = P[0] - ( lineCenter0 + (lineUnit0*nodeIt) );
		posMag += POW2(posDir0);
		negDir0 = P[0] - ( lineCenter0 - (lineUnit0*nodeIt) );
		negMag += POW2(negDir0);

		posDir1 = P[1] - ( lineCenter1 + (lineUnit1*nodeIt) );
		posMag += POW2(posDir1);
		negDir1 = P[1] - ( lineCenter1 - (lineUnit1*nodeIt));
		negMag += POW2(negDir1);

		posDir2 = P[2] - ( lineCenter2 + (lineUnit2*nodeIt) );
		posMag += POW2(posDir2);
		negDir2 = P[2] - ( lineCenter2 - (lineUnit2*nodeIt) );
		negMag += POW2(negDir2);

		posMag = 1. / SQRT( posMag );
		negMag = 1. / SQRT( negMag );

		sum += weightFac * (posMag + negMag);
	}


	const CU_TYPE oneOverEps0 = 1./(M_EPS0);

    return (sum*data[6]*0.25*oneOverEps0);
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE4 EL_EField_Quad16N(const CU_TYPE* P, const CU_TYPE* data)
{
	const CU_TYPE lineCenter0 = (data[0] + data[3])*.5;
	const CU_TYPE lineCenter1 = (data[1] + data[4])*.5;
	const CU_TYPE lineCenter2 = (data[2] + data[5])*.5;

	CU_TYPE lineLength[1];
	Line_Length( lineLength, data );
	CU_TYPE halfLength = 0.5 * lineLength[0]; // only half of length is needed in iteration

    CU_TYPE prefacUnit = 1./lineLength[0];
    CU_TYPE lineUnit0 = prefacUnit * (data[3] - data[0]);
    CU_TYPE lineUnit1 = prefacUnit * (data[4] - data[1]);
    CU_TYPE lineUnit2 = prefacUnit * (data[5] - data[2]);

    CU_TYPE sum0 = 0.;
    CU_TYPE sum1 = 0.;
    CU_TYPE sum2 = 0.;

    CU_TYPE posDir0 = 0.;
    CU_TYPE posDir1 = 0.;
    CU_TYPE posDir2 = 0.;

    CU_TYPE negDir0 = 0.;
    CU_TYPE negDir1 = 0.;
    CU_TYPE negDir2 = 0.;

    CU_TYPE posMag = 0.;
    CU_TYPE negMag = 0.;

	CU_TYPE weightFac = 0.;
	CU_TYPE nodeIt = 0.;

	for( unsigned short i=0; i<8; i++ ) {
		weightFac = cuLineQuadw16[i]*halfLength;
		nodeIt = cuLineQuadx16[i]*halfLength;

		// reset variables for magnitude
		posMag = 0.;
		negMag = 0.;

		posDir0 = P[0] - ( lineCenter0 + (lineUnit0*nodeIt) );
		posDir1 = P[1] - ( lineCenter1 + (lineUnit1*nodeIt) );
		posDir2 = P[2] - ( lineCenter2 + (lineUnit2*nodeIt) );

		negDir0 = P[0] - ( lineCenter0 - (lineUnit0*nodeIt) );
		negDir1 = P[1] - ( lineCenter1 - (lineUnit1*nodeIt) );
		negDir2 = P[2] - ( lineCenter2 - (lineUnit2*nodeIt) );

		posMag = POW2(posDir0) + POW2(posDir1) + POW2(posDir2);
		posMag = 1. / SQRT( posMag );
		posMag = POW3( posMag );

		negMag = POW2(negDir0) + POW2(negDir1) + POW2(negDir2);
		negMag = 1. / SQRT( negMag );
		negMag = POW3( negMag );

		sum0 += weightFac * (posMag*posDir0 + negMag*negDir0);
		sum1 += weightFac * (posMag*posDir1 + negMag*negDir1);
		sum2 += weightFac * (posMag*posDir2 + negMag*negDir2);
	}

	const CU_TYPE prefac = (0.25*data[6])/(M_EPS0);

    return MAKECU4( prefac*sum0, prefac*sum1, prefac*sum2, 0. );
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE4 EL_EFieldAndPotential_Quad16N(const CU_TYPE* P, const CU_TYPE* data)
{
	const CU_TYPE lineCenter0 = (data[0] + data[3])*.5;
	const CU_TYPE lineCenter1 = (data[1] + data[4])*.5;
	const CU_TYPE lineCenter2 = (data[2] + data[5])*.5;

	CU_TYPE lineLength[1];
	Line_Length( lineLength, data );
	CU_TYPE halfLength = 0.5 * lineLength[0]; // only half of length is needed in iteration

    CU_TYPE prefacUnit = 1./lineLength[0];
    CU_TYPE lineUnit0 = prefacUnit * (data[3] - data[0]);
    CU_TYPE lineUnit1 = prefacUnit * (data[4] - data[1]);
    CU_TYPE lineUnit2 = prefacUnit * (data[5] - data[2]);

	CU_TYPE sum0 = 0.; /* field */
	CU_TYPE sum1 = 0.;
	CU_TYPE sum2 = 0.;
	CU_TYPE sum3 = 0.; /* potential */

    CU_TYPE posDir0 = 0.;
    CU_TYPE posDir1 = 0.;
    CU_TYPE posDir2 = 0.;

    CU_TYPE negDir0 = 0.;
    CU_TYPE negDir1 = 0.;
    CU_TYPE negDir2 = 0.;

    CU_TYPE posMag = 0.;
    CU_TYPE negMag = 0.;

	CU_TYPE weightFac = 0.;
	CU_TYPE nodeIt = 0.;

	for( unsigned short i=0; i<8; i++ ) {
		weightFac = cuLineQuadw16[i]*halfLength;
		nodeIt = cuLineQuadx16[i]*halfLength;

		// reset variables for magnitude
		posMag = 0.;
		negMag = 0.;

		posDir0 = P[0] - ( lineCenter0 + (lineUnit0*nodeIt) );
		posDir1 = P[1] - ( lineCenter1 + (lineUnit1*nodeIt) );
		posDir2 = P[2] - ( lineCenter2 + (lineUnit2*nodeIt) );

		negDir0 = P[0] - ( lineCenter0 - (lineUnit0*nodeIt) );
		negDir1 = P[1] - ( lineCenter1 - (lineUnit1*nodeIt) );
		negDir2 = P[2] - ( lineCenter2 - (lineUnit2*nodeIt) );

		posMag = POW2(posDir0) + POW2(posDir1) + POW2(posDir2);
		posMag = 1. / SQRT( posMag );

		negMag = POW2(negDir0) + POW2(negDir1) + POW2(negDir2);
		negMag = 1. / SQRT( negMag );

		sum3 += weightFac * (posMag + negMag);

		posMag = POW3( posMag );
		negMag = POW3( negMag );

		sum0 += weightFac * (posMag*posDir0 + negMag*negDir0);
		sum1 += weightFac * (posMag*posDir1 + negMag*negDir1);
		sum2 += weightFac * (posMag*posDir2 + negMag*negDir2);
	}

	const CU_TYPE prefac = (0.25*data[6])/(M_EPS0);

    return MAKECU4( prefac*sum0, prefac*sum1, prefac*sum2, prefac*sum3 );
}


#endif /* KEMFIELD_ELECTROSTATICQUADRATURELINESEGMENT_CUH */
