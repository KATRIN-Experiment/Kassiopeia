#ifndef KEMFIELD_ELECTROSTATICQUADRATURELINESEGMENT_CL
#define KEMFIELD_ELECTROSTATICQUADRATURELINESEGMENT_CL

// OpenCL kernel for line segment boundary integrator with 4-node and 16-node Gaussian quadrature
// Detailed information on the quadrature implementation can be found in the CPU code,
// class 'KElectrostaticQuadratureLineSegmentIntegrator'.
// Author: Daniel Hilk
//
// This kernel version is optimized regarding workgroup size and speed for compute
// devices providing only scalar units, but runs as well very efficient on devices
// providing vector units additionally. The optimal sizes for the used hardware
// will be set automatically.
//
// Recommended workgroup sizes by driver for NVIDIA Tesla K40c:
// * EL_Potential_Quad4N: 1024
// * EL_EField_Quad4N: 1024
// * EL_EFieldAndPotential_Quad4N: 1024
// * EL_Potential_Quad16N: 1024
// * EL_EField_Quad16N: 896
// * EL_EFieldAndPotential_Quad16N: 896

#include "kEMField_LineSegment.cl"

// Wire geometry definition (as defined by the streamers in KLineSegment.hh):
//
// data[0..2]: P0[0..2]
// data[3..5]: P1[0..2]
// data[6]:    diameter

//______________________________________________________________________________

    // Gaussian nodes and  weights

	__constant CL_TYPE oclLineQuadx4[2] = {0.339981043584856, 0.861136311594053};
	__constant CL_TYPE oclLineQuadw4[2] = {0.652145154862546, 0.347854845137454};

	__constant CL_TYPE oclLineQuadx16[8] = {0.09501250983763744, 0.28160355077925891, 0.45801677765722739, 0.61787624440264375,
			0.75540440835500303, 0.86563120238783174, 0.94457502307323258, 0.98940093499164993};
	__constant CL_TYPE oclLineQuadw16[8] = {0.189450610455068496, 0.182603415044923589, 0.169156519395002532,
			0.149595988816576731,
			0.124628971255533872, 0.095158511682492785, 0.062253523938647892,
			0.027152459411754095};

//______________________________________________________________________________

CL_TYPE EL_Potential_Quad4N( const CL_TYPE* P, __global const CL_TYPE* data )
{
	const CL_TYPE lineCenter0 = (data[0] + data[3])*.5;
	const CL_TYPE lineCenter1 = (data[1] + data[4])*.5;
	const CL_TYPE lineCenter2 = (data[2] + data[5])*.5;

	CL_TYPE lineLength[1];
	Line_Length( lineLength, data );
	const CL_TYPE halfLength = 0.5 * lineLength[0]; // only half of length is needed in iteration

	const CL_TYPE prefacUnit = 1./lineLength[0];
	const CL_TYPE lineUnit0 = prefacUnit * (data[3] - data[0]);
	const CL_TYPE lineUnit1 = prefacUnit * (data[4] - data[1]);
	const CL_TYPE lineUnit2 = prefacUnit * (data[5] - data[2]);

	CL_TYPE sum = 0.;

	CL_TYPE posDir0 = 0.;
	CL_TYPE posDir1 = 0.;
	CL_TYPE posDir2 = 0.;

	CL_TYPE negDir0 = 0.;
	CL_TYPE negDir1 = 0.;
	CL_TYPE negDir2 = 0.;

	CL_TYPE posMag = 0.;
	CL_TYPE negMag = 0.;

	CL_TYPE weightFac = 0.;
	CL_TYPE nodeIt = 0.;

	for( unsigned short i=0; i<2; i++ ) {
		weightFac = oclLineQuadw4[i] * halfLength;
		nodeIt = oclLineQuadx4[i] * halfLength;

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

	const CL_TYPE oneOverEps0 = 1./(M_EPS0);

	return (sum*data[6]*0.25*oneOverEps0);
}

//______________________________________________________________________________

CL_TYPE4 EL_EField_Quad4N(const CL_TYPE* P, __global const CL_TYPE* data)
{
	const CL_TYPE lineCenter0 = (data[0] + data[3])*.5;
	const CL_TYPE lineCenter1 = (data[1] + data[4])*.5;
	const CL_TYPE lineCenter2 = (data[2] + data[5])*.5;

	CL_TYPE lineLength[1];
	Line_Length( lineLength, data );
	const CL_TYPE halfLength = 0.5 * lineLength[0]; // only half of length is needed in iteration

	const CL_TYPE prefacUnit = 1./lineLength[0];
	const CL_TYPE lineUnit0 = prefacUnit * (data[3] - data[0]);
	const CL_TYPE lineUnit1 = prefacUnit * (data[4] - data[1]);
	const CL_TYPE lineUnit2 = prefacUnit * (data[5] - data[2]);

	CL_TYPE sum0 = 0.;
	CL_TYPE sum1 = 0.;
	CL_TYPE sum2 = 0.;

	CL_TYPE posDir0 = 0.;
	CL_TYPE posDir1 = 0.;
	CL_TYPE posDir2 = 0.;

	CL_TYPE negDir0 = 0.;
	CL_TYPE negDir1 = 0.;
	CL_TYPE negDir2 = 0.;

	CL_TYPE posMag = 0.;
	CL_TYPE negMag = 0.;

	CL_TYPE weightFac = 0.;
	CL_TYPE nodeIt = 0.;

	for( unsigned short i=0; i<2; i++ ) {
		weightFac = oclLineQuadw4[i]*halfLength;
		nodeIt = oclLineQuadx4[i]*halfLength;

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

	const CL_TYPE prefac = (0.25*data[6])/(M_EPS0);

	return (CL_TYPE4)( prefac*sum0, prefac*sum1, prefac*sum2, 0. );
}

//______________________________________________________________________________

CL_TYPE4 EL_EFieldAndPotential_Quad4N(const CL_TYPE* P, __global const CL_TYPE* data)
{
	const CL_TYPE lineCenter0 = (data[0] + data[3])*.5;
	const CL_TYPE lineCenter1 = (data[1] + data[4])*.5;
	const CL_TYPE lineCenter2 = (data[2] + data[5])*.5;

	CL_TYPE lineLength[1];
	Line_Length( lineLength, data );
	const CL_TYPE halfLength = 0.5 * lineLength[0]; // only half of length is needed in iteration

	const CL_TYPE prefacUnit = 1./lineLength[0];
	const CL_TYPE lineUnit0 = prefacUnit * (data[3] - data[0]);
	const CL_TYPE lineUnit1 = prefacUnit * (data[4] - data[1]);
	const CL_TYPE lineUnit2 = prefacUnit * (data[5] - data[2]);

	CL_TYPE sum0 = 0.; /* field */
	CL_TYPE sum1 = 0.;
	CL_TYPE sum2 = 0.;
	CL_TYPE sum3 = 0.; /* potential */

	CL_TYPE posDir0 = 0.;
	CL_TYPE posDir1 = 0.;
	CL_TYPE posDir2 = 0.;

	CL_TYPE negDir0 = 0.;
	CL_TYPE negDir1 = 0.;
	CL_TYPE negDir2 = 0.;

	CL_TYPE posMag = 0.;
	CL_TYPE negMag = 0.;

	CL_TYPE weightFac = 0.;
	CL_TYPE nodeIt = 0.;

	for( unsigned short i=0; i<2; i++ ) {
		weightFac = oclLineQuadw4[i]*halfLength;
		nodeIt = oclLineQuadx4[i]*halfLength;

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

	const CL_TYPE prefac = (0.25*data[6])/(M_EPS0);

	return (CL_TYPE4)( prefac*sum0, prefac*sum1, prefac*sum2, prefac*sum3 );
}

//______________________________________________________________________________

CL_TYPE EL_Potential_Quad16N( const CL_TYPE* P, __global const CL_TYPE* data )
{
	const CL_TYPE lineCenter0 = (data[0] + data[3])*.5;
	const CL_TYPE lineCenter1 = (data[1] + data[4])*.5;
	const CL_TYPE lineCenter2 = (data[2] + data[5])*.5;

	CL_TYPE lineLength[1];
	Line_Length( lineLength, data );
	const CL_TYPE halfLength = 0.5 * lineLength[0]; // only half of length is needed in iteration

	const CL_TYPE prefacUnit = 1./lineLength[0];
	const CL_TYPE lineUnit0 = prefacUnit * (data[3] - data[0]);
	const CL_TYPE lineUnit1 = prefacUnit * (data[4] - data[1]);
	const CL_TYPE lineUnit2 = prefacUnit * (data[5] - data[2]);

	CL_TYPE sum = 0.;

	CL_TYPE posDir0 = 0.;
	CL_TYPE posDir1 = 0.;
	CL_TYPE posDir2 = 0.;

	CL_TYPE negDir0 = 0.;
	CL_TYPE negDir1 = 0.;
	CL_TYPE negDir2 = 0.;

	CL_TYPE posMag = 0.;
	CL_TYPE negMag = 0.;

	CL_TYPE weightFac = 0.;
	CL_TYPE nodeIt = 0.;

	for( unsigned short i=0; i<8; i++ ) {
		weightFac = oclLineQuadw16[i] * halfLength;
		nodeIt = oclLineQuadx16[i] * halfLength;

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

	const CL_TYPE oneOverEps0 = 1./(M_EPS0);

	return (sum*data[6]*0.25*oneOverEps0);
}

//______________________________________________________________________________

CL_TYPE4 EL_EField_Quad16N(const CL_TYPE* P, __global const CL_TYPE* data)
{
	const CL_TYPE lineCenter0 = (data[0] + data[3])*.5;
	const CL_TYPE lineCenter1 = (data[1] + data[4])*.5;
	const CL_TYPE lineCenter2 = (data[2] + data[5])*.5;

	CL_TYPE lineLength[1];
	Line_Length( lineLength, data );
	const CL_TYPE halfLength = 0.5 * lineLength[0]; // only half of length is needed in iteration

	const CL_TYPE prefacUnit = 1./lineLength[0];
	const CL_TYPE lineUnit0 = prefacUnit * (data[3] - data[0]);
	const CL_TYPE lineUnit1 = prefacUnit * (data[4] - data[1]);
	const CL_TYPE lineUnit2 = prefacUnit * (data[5] - data[2]);

	CL_TYPE sum0 = 0.;
	CL_TYPE sum1 = 0.;
	CL_TYPE sum2 = 0.;

	CL_TYPE posDir0 = 0.;
	CL_TYPE posDir1 = 0.;
	CL_TYPE posDir2 = 0.;

	CL_TYPE negDir0 = 0.;
	CL_TYPE negDir1 = 0.;
	CL_TYPE negDir2 = 0.;

	CL_TYPE posMag = 0.;
	CL_TYPE negMag = 0.;

	CL_TYPE weightFac = 0.;
	CL_TYPE nodeIt = 0.;

	for( unsigned short i=0; i<8; i++ ) {
		weightFac = oclLineQuadw16[i]*halfLength;
		nodeIt = oclLineQuadx16[i]*halfLength;

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

	const CL_TYPE prefac = (0.25*data[6])/(M_EPS0);

	return (CL_TYPE4)( prefac*sum0, prefac*sum1, prefac*sum2, 0. );
}

//______________________________________________________________________________

CL_TYPE4 EL_EFieldAndPotential_Quad16N(const CL_TYPE* P, __global const CL_TYPE* data)
{
	const CL_TYPE lineCenter0 = (data[0] + data[3])*.5;
	const CL_TYPE lineCenter1 = (data[1] + data[4])*.5;
	const CL_TYPE lineCenter2 = (data[2] + data[5])*.5;

	CL_TYPE lineLength[1];
	Line_Length( lineLength, data );
	const CL_TYPE halfLength = 0.5 * lineLength[0]; // only half of length is needed in iteration

	const CL_TYPE prefacUnit = 1./lineLength[0];
	const CL_TYPE lineUnit0 = prefacUnit * (data[3] - data[0]);
	const CL_TYPE lineUnit1 = prefacUnit * (data[4] - data[1]);
	const CL_TYPE lineUnit2 = prefacUnit * (data[5] - data[2]);

	CL_TYPE sum0 = 0.; /* field */
	CL_TYPE sum1 = 0.;
	CL_TYPE sum2 = 0.;
	CL_TYPE sum3 = 0.; /* potential */

	CL_TYPE posDir0 = 0.;
	CL_TYPE posDir1 = 0.;
	CL_TYPE posDir2 = 0.;

	CL_TYPE negDir0 = 0.;
	CL_TYPE negDir1 = 0.;
	CL_TYPE negDir2 = 0.;

	CL_TYPE posMag = 0.;
	CL_TYPE negMag = 0.;

	CL_TYPE weightFac = 0.;
	CL_TYPE nodeIt = 0.;

	for( unsigned short i=0; i<8; i++ ) {
		weightFac = oclLineQuadw16[i]*halfLength;
		nodeIt = oclLineQuadx16[i]*halfLength;

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

	const CL_TYPE prefac = (0.25*data[6])/(M_EPS0);

	return (CL_TYPE4)( prefac*sum0, prefac*sum1, prefac*sum2, prefac*sum3 );
}


#endif /* KEMFIELD_ELECTROSTATICQUADRATURELINESEGMENT_CL */
