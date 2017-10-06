#include "KElectrostaticCubatureRectangleIntegrator.hh"

#define POW2(x) ((x)*(x))
#define POW3(x) ((x)*(x)*(x))


namespace KEMField
{
void KElectrostaticCubatureRectangleIntegrator::GaussPoints_Rect4P( const double* data, double* Q ) const
{
	// Calculates the 4 Gaussian points Q[0],...,Q[3]  for the 4-point cubature of the rectangle
	// P0, P2, P2: corner points of the rectangle
	// N1: unit vector from P0 to P1
	// N2: unit vector from P0 to P2
	// a: P0-P1 side length
	// b: P0-P2 side length

	const double c = 1./sqrt(3.);

	const double x[4] = {c, c, -c, -c};
	const double y[4] = {c, -c, c, -c};

	const double a2 = 0.5*data[0];
	const double b2 = 0.5*data[1];

	const double Qcen[3] = { data[2] + a2*data[5] + b2*data[8],
			data[3] + a2*data[6] + b2*data[9],
			data[4] + a2*data[7] + b2*data[10] };

	unsigned short index = 0;

	for( unsigned short i=0; i<4; i++) {
		// loop over components
		for( unsigned short j=0; j<3; j++ ) {
			index = (3*i) + j;
			Q[index] = Qcen[j] + (x[i]*a2)*data[5+j] + (y[i]*b2)*data[8+j];
		}
	}

	return;
}

void KElectrostaticCubatureRectangleIntegrator::GaussPoints_Rect7P( const double* data, double* Q ) const
{
	// Calculates the 7 Gaussian points Q[0],...,Q[6]  for the 7-point cubature of the rectangle
	// P0, P2, P2: corner points of the rectangle
	// N1: unit vector from P0 to P1
	// N2: unit vector from P0 to P2
	// a: P0-P1 side length
	// b: P0-P2 side length

	// GRECTCUB7INDEX1 defined     :  5-1 7-point formula on p. 246 of Stroud book
	// GRECTCUB7INDEX1 not defined :  5-2 7-point formula on p. 247 of Stroud book
	//  If Index7==1:  t1,r1,s1, x1, y1;
	//  If Index7==2:  t2,r2,s2, x2, y2;

#ifdef GRECTCUB7INDEX1
	const double t1 = sqrt(14./15.);
	const double r1 = sqrt(3./5.);
	const double s1 = sqrt(1./3.);

	const double x1[7] = {0., 0., 0., r1, r1, -r1, -r1};
	const double y1[7] = {0., t1, -t1, s1, -s1, s1, -s1};
#else
	const double t2 = sqrt((7.-sqrt(24.))/15.);
	const double s2 = sqrt((7.+sqrt(24.))/15.);
	const double r2 = sqrt(7./15.);

	const double x2[7] = {0., r2, -r2, s2, -s2, t2, -t2};
	const double y2[7] = {0., r2, -r2, -t2, t2, -s2, s2};
#endif

	const double a2 = 0.5*data[0];
	const double b2 = 0.5*data[1];

	const double Qcen[3] = { data[2] + a2*data[5] + b2*data[8],
			data[3] + a2*data[6] + b2*data[9],
			data[4] + a2*data[7] + b2*data[10] };

	unsigned short index = 0;

	for( unsigned short i=0; i<7; i++ ) {
		// loop over components
		for( unsigned short j=0; j<3; j++ ) {
			index = (3*i) + j;
#ifdef GRECTCUB7INDEX1
			Q[index] = Qcen[j] + (x1[i]*a2)*data[5+j] + (y1[i]*b2)*data[8+j];
#else
			Q[index] = Qcen[j] + (x2[i]*a2)*data[5+j] + (y2[i]*b2)*data[8+j];
#endif
		}
	}

	return;
}

void KElectrostaticCubatureRectangleIntegrator::GaussPoints_Rect9P( const double* data, double* Q ) const
{
	const double cx = 0.5*data[0];
	const double cy = 0.5*data[1];
	const double kx = gRectCub9term1[0]*cx;
	const double ky = gRectCub9term1[0]*cy;

	// loop over components
	for( unsigned short i=0; i<3; i++ )
	{
		Q[i] = data[2+i] + cx*data[5+i] + cy*data[8+i];
		Q[3+i] = Q[i] + kx*data[5+i];
		Q[6+i] = Q[i] - kx*data[5+i];
		Q[9+i] = Q[i] + ky*data[8+i];
		Q[12+i] = Q[i] - ky*data[8+i];
		Q[15+i] = Q[i] + kx*data[5+i] + ky*data[8+i];
		Q[18+i] = Q[i] + kx*data[5+i] - ky*data[8+i];
		Q[21+i] = Q[i] - kx*data[5+i] + ky*data[8+i];
		Q[24+i] = Q[i] - kx*data[5+i] - ky*data[8+i];
	}

	return;
}

void KElectrostaticCubatureRectangleIntegrator::GaussPoints_Rect12P( const double* data, double* Q ) const
{
	// Calculates the 12 Gaussian points Q[0],...,Q[11]  for the 12-point cubature of the rectangle;
	// See: Stroud book, page 253;  Tyler 1953, page 403.
	// P0, P2, P2: corner points of the rectangle
	// N1: unit vector from P0 to P1
	// N2: unit vector from P0 to P2
	// a: P0-P1 side length
	// b: P0-P2 side length

	const double r = sqrt(6./7.);
	const double s = sqrt((114.-3.*sqrt(583.))/287.);
	const double t = sqrt((114.+3.*sqrt(583.))/287.);

	const double x[12] = {r,  -r,  0., 0.,  s,  s,  -s,  -s,  t,  t,  -t,  -t   };
	const double y[12] = {0., 0.,  r,   -r,  s,  -s,  s,  -s,   t,  -t,  t,  -t    };

	const double a2 = 0.5*data[0];
	const double b2 = 0.5*data[1];

	const double Qcen[3] = { data[2] + a2*data[5] + b2*data[8],
			data[3] + a2*data[6] + b2*data[9],
			data[4] + a2*data[7] + b2*data[10] };

	unsigned short index = 0;

	for( unsigned short i=0; i<12; i++ ) {
		// loop over components
		for( unsigned short j=0; j<3; j++ ) {
			index = (3*i) + j;
			Q[index] = Qcen[j] + (x[i]*a2)*data[5+j] + (y[i]*b2)*data[8+j];
		}
	}

	return;
}

void KElectrostaticCubatureRectangleIntegrator::GaussPoints_Rect17P( const double* data, double* Q ) const
{
	// Calculates the 17 Gaussian points Q[0],...,Q[16]  for the 17-point cubature of the rectangle;
	// See: Engels book, page 257;  Möller 1976, page 194.
	// P0, P2, P2: corner points of the rectangle
	// N1: unit vector from P0 to P1
	// N2: unit vector from P0 to P2
	// a: P0-P1 side length
	// b: P0-P2 side length

	const double x1=0.96884996636197772072;
	const double y1=0.63068011973166885417;
	const double x2=0.75027709997890053354;
	const double y2=0.92796164595956966740;
	const double x3=0.52373582021442933604;
	const double y3=0.45333982113564719076;
	const double x4=0.07620832819261717318;
	const double y4=0.85261572933366230775;

	const double x[17]={0.,x1,-x1,-y1,y1,x2,-x2,-y2,y2,x3,-x3,-y3,y3,x4,-x4,-y4,y4};
	const double y[17]={0.,y1,-y1,x1,-x1,y2,-y2,x2,-x2,y3,-y3,x3,-x3,y4,-y4,x4,-x4};

	const double a2 = 0.5*data[0];
	const double b2 = 0.5*data[1];

	const double Qcen[3] = { data[2] + a2*data[5] + b2*data[8],
			data[3] + a2*data[6] + b2*data[9],
			data[4] + a2*data[7] + b2*data[10] };

	unsigned short index = 0;

	for( unsigned short i=0; i<17; i++ ) {
		// loop over components
		for( unsigned short j=0; j<3; j++ ) {
			index = (3*i) + j;
			Q[index] = Qcen[j] + (x[i]*a2)*data[5+j] + (y[i]*b2)*data[8+j];
		}
	}

	return;
}

void KElectrostaticCubatureRectangleIntegrator::GaussPoints_Rect20P( const double* data, double* Q ) const
{
	const double cx = 0.5*data[0];
	const double cy = 0.5*data[1];

	const double cen[3] = { data[2] + 0.5*data[0]*data[5] + 0.5*data[1]*data[8],
			data[3] + 0.5*data[0]*data[6] + 0.5*data[1]*data[9],
			data[4] + 0.5*data[0]*data[7] + 0.5*data[1]*data[10]};

	// loop over components
	for( unsigned short i=0; i<3; i++ )
	{
		Q[i] = cen[i] + (cx*gRectCub20term1[0]*data[5+i]);
		Q[3+i] = cen[i] - (cx*gRectCub20term1[0]*data[5+i]);
		Q[6+i] = cen[i] + (cy*gRectCub20term1[0]*data[8+i]);
		Q[9+i] = cen[i] - (cy*gRectCub20term1[0]*data[8+i]);
		Q[12+i] = cen[i] + (cx*gRectCub20term2[0]*data[5+i]);
		Q[15+i] = cen[i] - (cx*gRectCub20term2[0]*data[5+i]);
		Q[18+i] = cen[i] + (cy*gRectCub20term2[0]*data[8+i]);
		Q[21+i] = cen[i] - (cy*gRectCub20term2[0]*data[8+i]);
		Q[24+i] = cen[i] + (cx*gRectCub20term3[0]*data[5+i]) + (cy*gRectCub20term3[0]*data[8+i]);
		Q[27+i] = cen[i] + (cx*gRectCub20term3[0]*data[5+i]) - (cy*gRectCub20term3[0]*data[8+i]);
		Q[30+i] = cen[i] - (cx*gRectCub20term3[0]*data[5+i]) + (cy*gRectCub20term3[0]*data[8+i]);
		Q[33+i] = cen[i] - (cx*gRectCub20term3[0]*data[5+i]) - (cy*gRectCub20term3[0]*data[8+i]);
		Q[36+i] = cen[i] + (cx*gRectCub20term4[0]*data[5+i]) + (cy*gRectCub20term5[0]*data[8+i]);
		Q[39+i]= cen[i] + (cx*gRectCub20term5[0]*data[5+i]) + (cy*gRectCub20term4[0]*data[8+i]);
		Q[42+i] = cen[i] + (cx*gRectCub20term4[0]*data[5+i]) - (cy*gRectCub20term5[0]*data[8+i]);
		Q[45+i] = cen[i] + (cx*gRectCub20term5[0]*data[5+i]) - (cy*gRectCub20term4[0]*data[8+i]);
		Q[48+i] = cen[i] - (cx*gRectCub20term4[0]*data[5+i]) + (cy*gRectCub20term5[0]*data[8+i]);
		Q[51+i] = cen[i] - (cx*gRectCub20term5[0]*data[5+i]) + (cy*gRectCub20term4[0]*data[8+i]);
		Q[54+i] = cen[i] - (cx*gRectCub20term4[0]*data[5+i]) - (cy*gRectCub20term5[0]*data[8+i]);
		Q[57+i] = cen[i] - (cx*gRectCub20term5[0]*data[5+i]) - (cy*gRectCub20term4[0]*data[8+i]);
	}

	return;
}

void KElectrostaticCubatureRectangleIntegrator::GaussPoints_Rect33P( const double* data, double* Q ) const
{
	// Calculates the 33 Gaussian points Q[0],...,Q[32]  for the 33-point cubature of the rectangle;
	// See: Cools, Haegemans, Computing  40 (1988) 139
	// P0, P2, P2: corner points of the rectangle
	// N1: unit vector from P0 to P1
	// N2: unit vector from P0 to P2
	// a: P0-P1 side length
	// b: P0-P2 side length

	const double X[9]={
			0.00000000000000000000,
			0.77880971155441942252,
			0.95729769978630736566,
			0.13818345986246535375,
			0.94132722587292523695,
			0.47580862521827590507,
			0.75580535657208143627,
			0.69625007849174941396,
			0.34271655604040678941
	};
	const double Y[9]={
			0.00000000000000000000,
			0.98348668243987226379,
			0.85955600564163892859,
			0.95892517028753485754,
			0.39073621612946100068,
			0.85007667369974857597,
			0.64782163718701073204,
			0.70741508996444936217e-1,
			0.40930456169403884330
	};

	const double x[33]={
			X[0],
			X[1], -X[1],
			-Y[1], Y[1],
			X[2], -X[2],
			-Y[2], Y[2],
			X[3], -X[3],
			-Y[3], Y[3],
			X[4], -X[4],
			-Y[4], Y[4],
			X[5], -X[5],
			-Y[5], Y[5],
			X[6], -X[6],
			-Y[6], Y[6],
			X[7], -X[7],
			-Y[7], Y[7],
			X[8], -X[8],
			-Y[8], Y[8] };

	const double y[33]={
			Y[0],
			Y[1], -Y[1],
			X[1], -X[1],
			Y[2], -Y[2],
			X[2], -X[2],
			Y[3], -Y[3],
			X[3], -X[3],
			Y[4], -Y[4],
			X[4], -X[4],
			Y[5], -Y[5],
			X[5], -X[5],
			Y[6], -Y[6],
			X[6], -X[6],
			Y[7], -Y[7],
			X[7], -X[7],
			Y[8], -Y[8],
			X[8], -X[8] };

	const double a2 = 0.5*data[0];
	const double b2 = 0.5*data[1];

	const double Qcen[3] = { data[2] + a2*data[5] + b2*data[8],
			data[3] + a2*data[6] + b2*data[9],
			data[4] + a2*data[7] + b2*data[10] };

	unsigned short index = 0;

	for(int i=0; i<=32; i++) {
		// loop over components
		for( unsigned short j=0; j<3; j++ ) {
			index = (3*i) + j;
			Q[index] = Qcen[j] + (x[i]*a2)*data[5+j] + (y[i]*b2)*data[8+j];
		}
	}

	return;
}

double KElectrostaticCubatureRectangleIntegrator::Potential_RectNP( const double* data, const KPosition& P, const unsigned short noPoints, double* Q, const double* weights ) const
{
	const double area = data[0] * data[1];
	const double prefac = area * KEMConstants::OneOverFourPiEps0;

	double partialValue( 0. );
	double distVector[3] = {0., 0., 0.};
	double finalSum( 0. );
	double oneOverAbsoluteValue( 0. );
	double absoluteValueSquared( 0. );
	double componentSquared( 0. );
	unsigned short arrayIndex( 0 );

	// loop over Gaussian points
	for( unsigned short gaussianPoint=0; gaussianPoint<noPoints; gaussianPoint++ )
	{
		absoluteValueSquared = 0.;

		// loop over vector components
		for( unsigned short componentIndex=0; componentIndex<3; componentIndex++ )
		{
			// getting index of Gaussian point array Q[3*noPoints]
			arrayIndex = ((3*gaussianPoint) + componentIndex);

			// component of distance vector from Gaussian point to computation point ( P - Q_i )
			distVector[componentIndex] = ( P[componentIndex] - Q[arrayIndex] );

			// square current vector component
			componentSquared = POW2(distVector[componentIndex]);

			// sum up to variable absoluteValueSquared
			absoluteValueSquared += componentSquared;
		} /* components */

		// separate divisions from computation, here: tmp2 = 1/|P-Q_i|
		oneOverAbsoluteValue = 1./sqrt(absoluteValueSquared);

		// partialValue = w_i * (1/|P-Q_i|)
		partialValue = weights[gaussianPoint]*oneOverAbsoluteValue;

		// sum up for final result
		finalSum += partialValue;
	} /* Gaussian points */

	return (prefac*finalSum);
}

KEMThreeVector KElectrostaticCubatureRectangleIntegrator::ElectricField_RectNP( const double* data, const KPosition& P, const unsigned short noPoints, double* Q, const double* weights ) const
{
	const double area = data[0] * data[1];
	const double prefac = area * KEMConstants::OneOverFourPiEps0;

	double partialValue[3] = {0., 0., 0.};
	double distVector[3] = {0., 0., 0.};
	double finalSum[3] = {0., 0., 0.};
	double oneOverAbsoluteValue( 0. );
	double tmpValue( 0. );
	double absoluteValueSquared( 0. ) ;
	double componentSquared( 0. );
	unsigned short arrayIndex( 0 );

	// loop over Gaussian points
	for( unsigned short gaussianPoint=0; gaussianPoint<noPoints; gaussianPoint++ )
	{
		absoluteValueSquared = 0.;

		// loop over vector components
		for( unsigned short componentIndex=0; componentIndex<3; componentIndex++ )
		{
			// getting index of Gaussian point array Q[3*noPoints]
			arrayIndex = ((3*gaussianPoint) + componentIndex);

			// component of distance vector from Gaussian point to computation point ( P - Q_i )
			distVector[componentIndex] = ( P[componentIndex] - Q[arrayIndex] );

			// square current vector component
			componentSquared = POW2(distVector[componentIndex]);

			// sum up to variable absoluteValueSquared
			absoluteValueSquared += componentSquared;

			// partialValue = P-Q_i
			partialValue[componentIndex] = distVector[componentIndex];
		} /* components */

		// separate divisions from computation, here: tmp2 = 1/|P-Q_i|
		oneOverAbsoluteValue = 1./sqrt(absoluteValueSquared);
		tmpValue = POW3(oneOverAbsoluteValue);
		tmpValue = weights[gaussianPoint] * tmpValue;

		// partialValue = partialValue * w_i * (1/|P-Q_i|^3)
		partialValue[0] *= tmpValue;
		partialValue[1] *= tmpValue;
		partialValue[2] *= tmpValue;

		// sum up for final result
		finalSum[0] += partialValue[0];
		finalSum[1] += partialValue[1];
		finalSum[2] += partialValue[2];
	} /* Gaussian points */

	for( unsigned short i=0; i<3; i++ ){
		finalSum[i] = prefac * finalSum[i];
	}

	return KEMThreeVector( finalSum[0], finalSum[1], finalSum[2] );
}

std::pair<KEMThreeVector,double> KElectrostaticCubatureRectangleIntegrator::ElectricFieldAndPotential_RectNP( const double* data, const KPosition& P, const unsigned short noPoints, double* Q, const double* weights ) const
{
	const double area = data[0] * data[1];
	const double prefac = area * KEMConstants::OneOverFourPiEps0;

	double partialValue[4] = {0., 0., 0., 0.};
	double distVector[3] = {0., 0., 0.};
	double finalSum[4] = {0., 0., 0., 0.};
	double oneOverAbsoluteValue( 0. );
	double tmpValue( 0. );
	double absoluteValueSquared( 0. ) ;
	double componentSquared( 0. );
	unsigned short arrayIndex( 0 );

	// loop over Gaussian points
	for( unsigned short gaussianPoint=0; gaussianPoint<noPoints; gaussianPoint++ )
	{
		absoluteValueSquared = 0.;

		// loop over vector components
		for( unsigned short componentIndex=0; componentIndex<3; componentIndex++ )
		{
			// getting index of Gaussian point array Q[3*noPoints]
			arrayIndex = ((3*gaussianPoint) + componentIndex);

			// component of distance vector from Gaussian point to computation point ( P - Q_i )
			distVector[componentIndex] = ( P[componentIndex] - Q[arrayIndex] );

			// square current vector component
			componentSquared = POW2(distVector[componentIndex]);

			// sum up to variable absoluteValueSquared
			absoluteValueSquared += componentSquared;

			// partialValue = P-Q_i
			partialValue[componentIndex] = distVector[componentIndex];
		} /* components */

		// separate divisions from computation, here: tmp2 = 1/|P-Q_i|
		oneOverAbsoluteValue = 1./sqrt(absoluteValueSquared);
		tmpValue = POW3(oneOverAbsoluteValue);
		tmpValue = tmpValue*weights[gaussianPoint];

		// partialValue = partialValue (= vec r) * w_i * (1/|P-Q_i|^3)
		partialValue[0] *= tmpValue;
		partialValue[1] *= tmpValue;
		partialValue[2] *= tmpValue;
		partialValue[3] = weights[gaussianPoint]*oneOverAbsoluteValue;

		// sum up for final result
		finalSum[0] += partialValue[0];
		finalSum[1] += partialValue[1];
		finalSum[2] += partialValue[2];
		finalSum[3] += partialValue[3];
	} /* Gaussian points */

	for( unsigned short i=0; i<4; i++ ) {
		finalSum[i] = prefac*finalSum[i];
	}

	return std::make_pair( KEMThreeVector(finalSum[0],finalSum[1],finalSum[2]), finalSum[3] );
}

double KElectrostaticCubatureRectangleIntegrator::Potential(const KRectangle* source, const KPosition& P) const
{
	// save geometry info on rectangle into array, same convention as OpenCL implementation

	const double data[11] = {
			source->GetA(),
			source->GetB(),
			source->GetP0().X(),
			source->GetP0().Y(),
			source->GetP0().Z(),
			source->GetN1().X(),
			source->GetN1().Y(),
			source->GetN1().Z(),
			source->GetN2().X(),
			source->GetN2().Y(),
			source->GetN2().Z()};

    const double size = 0.5*(data[0]+data[1]);

	// get distance vector from field point to centroid of rectangle

    const double cen[3] = {
    		data[2] + 0.5*(data[0]*data[5] + data[1]*data[8]),
			data[3] + 0.5*(data[0]*data[6] + data[1]*data[9]),
			data[4] + 0.5*(data[0]*data[7] + data[1]*data[10])};

	const double dist[3] = { cen[0]-P[0], cen[1]-P[1], cen[2]-P[2] };

	// magnitude of distance vector
	const double mag = sqrt(POW2(dist[0]) + POW2(dist[1]) + POW2(dist[2]));

	// determine distance ratio (distance of field point to centroid over average triangle side length)
	const double tmpSize = 1./size;
	const double distanceRatio = mag*tmpSize;

    if( distanceRatio > fDrCutOffCub12 ) {
        // compute Gaussian points
        double rectQ7[21];
        GaussPoints_Rect7P(data,rectQ7);

        return Potential_RectNP( data, P, 7, rectQ7, gRectCub7w );
    }
    if( distanceRatio > fDrCutOffCub33 ) {
        // compute Gaussian points
        double rectQ12[36];
        GaussPoints_Rect12P(data,rectQ12);

        return Potential_RectNP( data, P, 12, rectQ12, gRectCub12w );
    }
    if( distanceRatio > fDrCutOffRWG ) {
        // compute Gaussian points
        double rectQ33[99];
        GaussPoints_Rect33P(data,rectQ33);

        return Potential_RectNP( data, P, 33, rectQ33, gRectCub33w );
    }

	return KElectrostaticRWGRectangleIntegrator::Potential( source, P );
}

KEMThreeVector KElectrostaticCubatureRectangleIntegrator::ElectricField(const KRectangle* source, const KPosition& P) const
{
	// save geometry info on rectangle into array, same convention as OpenCL implementation

	const double data[11] = {
			source->GetA(),
			source->GetB(),
			source->GetP0().X(),
			source->GetP0().Y(),
			source->GetP0().Z(),
			source->GetN1().X(),
			source->GetN1().Y(),
			source->GetN1().Z(),
			source->GetN2().X(),
			source->GetN2().Y(),
			source->GetN2().Z()};

    const double size = 0.5*(data[0]+data[1]);

	// get distance vector from field point to centroid of rectangle

    const double cen[3] = {
    		data[2] + 0.5*(data[0]*data[5] + data[1]*data[8]),
			data[3] + 0.5*(data[0]*data[6] + data[1]*data[9]),
			data[4] + 0.5*(data[0]*data[7] + data[1]*data[10])};

	const double dist[3] = { cen[0]-P[0], cen[1]-P[1], cen[2]-P[2] };

	// magnitude of distance vector
	const double mag = sqrt(POW2(dist[0]) + POW2(dist[1]) + POW2(dist[2]));

	// determine distance ratio (distance of field point to centroid over average triangle side length)
	const double tmpSize = 1./size;
	const double distanceRatio = mag*tmpSize;

    if( distanceRatio > fDrCutOffCub12 ) {
        // compute Gaussian points
        double rectQ7[21];
        GaussPoints_Rect7P(data,rectQ7);

        return ElectricField_RectNP( data, P, 7, rectQ7, gRectCub7w );
    }
    if( distanceRatio > fDrCutOffCub33 ) {
        // compute Gaussian points
        double rectQ12[36];
        GaussPoints_Rect12P(data,rectQ12);

        return ElectricField_RectNP( data, P, 12, rectQ12, gRectCub12w );
    }
    if( distanceRatio > fDrCutOffRWG ) {
        // compute Gaussian points
        double rectQ33[99];
        GaussPoints_Rect33P(data,rectQ33);

        return ElectricField_RectNP( data, P, 33, rectQ33, gRectCub33w );
    }

	return KElectrostaticRWGRectangleIntegrator::ElectricField( source, P );
}

std::pair<KEMThreeVector, double> KElectrostaticCubatureRectangleIntegrator::ElectricFieldAndPotential(const KRectangle* source, const KPosition& P) const
{
	// save geometry info on rectangle into array, same convention as OpenCL implementation

	const double data[11] = {
			source->GetA(),
			source->GetB(),
			source->GetP0().X(),
			source->GetP0().Y(),
			source->GetP0().Z(),
			source->GetN1().X(),
			source->GetN1().Y(),
			source->GetN1().Z(),
			source->GetN2().X(),
			source->GetN2().Y(),
			source->GetN2().Z()};

    const double size = 0.5*(data[0]+data[1]);

	// get distance vector from field point to centroid of rectangle

    const double cen[3] = {
    		data[2] + 0.5*(data[0]*data[5] + data[1]*data[8]),
			data[3] + 0.5*(data[0]*data[6] + data[1]*data[9]),
			data[4] + 0.5*(data[0]*data[7] + data[1]*data[10])};

	const double dist[3] = { cen[0]-P[0], cen[1]-P[1], cen[2]-P[2] };

	// magnitude of distance vector
	const double mag = sqrt(POW2(dist[0]) + POW2(dist[1]) + POW2(dist[2]));

	// determine distance ratio (distance of field point to centroid over average triangle side length)
	const double tmpSize = 1./size;
	const double distanceRatio = mag*tmpSize;

	if( distanceRatio > fDrCutOffCub12 ) {
		// compute Gaussian points
		double rectQ7[21];
		GaussPoints_Rect7P(data,rectQ7);

		return ElectricFieldAndPotential_RectNP( data, P, 7, rectQ7, gRectCub7w );
	}
    if( distanceRatio > fDrCutOffCub33 ) {
        // compute Gaussian points
        double rectQ12[36];
        GaussPoints_Rect12P(data,rectQ12);

        return ElectricFieldAndPotential_RectNP( data, P, 12, rectQ12, gRectCub12w );
    }
	if( distanceRatio > fDrCutOffRWG ) {
		// compute Gaussian points
		double rectQ33[99];
		GaussPoints_Rect33P(data,rectQ33);

		return ElectricFieldAndPotential_RectNP( data, P, 33, rectQ33, gRectCub33w );
	}

	return KElectrostaticRWGRectangleIntegrator::ElectricFieldAndPotential( source, P );
}

double KElectrostaticCubatureRectangleIntegrator::Potential(const KSymmetryGroup<KRectangle>* source, const KPosition& P) const
{
	double potential = 0.;
	for (KSymmetryGroup<KRectangle>::ShapeCIt it=source->begin();it!=source->end();++it)
		potential += Potential(*it,P);
	return potential;
}

KEMThreeVector KElectrostaticCubatureRectangleIntegrator::ElectricField(const KSymmetryGroup<KRectangle>* source, const KPosition& P) const
{
	KEMThreeVector electricField(0.,0.,0.);
	for (KSymmetryGroup<KRectangle>::ShapeCIt it=source->begin();it!=source->end();++it)
		electricField += ElectricField(*it,P);
	return electricField;
}

std::pair<KEMThreeVector, double> KElectrostaticCubatureRectangleIntegrator::ElectricFieldAndPotential(const KSymmetryGroup<KRectangle>* source, const KPosition& P) const
{
	std::pair<KEMThreeVector, double> fieldAndPotential;
	double potential( 0. );
	KEMThreeVector electricField( 0., 0., 0. );

	for( KSymmetryGroup<KRectangle>::ShapeCIt it=source->begin(); it!=source->end(); ++it ) {
		fieldAndPotential = ElectricFieldAndPotential( *it, P );
		electricField += fieldAndPotential.first;
		potential += fieldAndPotential.second;
	}

	return std::make_pair( electricField, potential );
}

}
