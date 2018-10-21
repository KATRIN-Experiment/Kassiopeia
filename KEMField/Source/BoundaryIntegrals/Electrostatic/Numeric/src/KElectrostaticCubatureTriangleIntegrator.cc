#include "KElectrostaticCubatureTriangleIntegrator.hh"

#include "KDataDisplay.hh"

#define POW2(x) ((x)*(x))
#define POW3(x) ((x)*(x)*(x))


namespace KEMField
{
void KElectrostaticCubatureTriangleIntegrator::GaussPoints_Tri4P( const double* data, double* Q ) const
{
	// Calculates the 4 Gaussian points Q[0],...,Q[3]  for the 4-point cubature of the triangle
	//  A=T[0], B=T[1], C=T[2]:  corner points of the triangle
	// alpha, beta, gamma: barycentric (area) coordinates of the Gaussian points;
	// alpha+beta+gamma=1;
	// Gaussian point is weighted average of the corner points A, B, C with the alpha, beta ,gamma weights

	const double A[3] = { data[2], data[3], data[4] };
	const double B[3] = { data[2] + (data[0]*data[5]),
			data[3] + (data[0]*data[6]),
			data[4] + (data[0]*data[7]) };
	const double C[3] = { data[2] + (data[1]*data[8]),
			data[3] + (data[1]*data[9]),
			data[4] + (data[1]*data[10]) };

	// loop over components
	for( unsigned short i=0; i<3; i++ ) {
		Q[i] = gTriCub4alpha[0]*A[i] + gTriCub4beta[0]*B[i] + gTriCub4gamma[0]*C[i];
		Q[i+3] = gTriCub4alpha[1]*A[i] + gTriCub4beta[1]*B[i] + gTriCub4gamma[1]*C[i];
		Q[i+6] = gTriCub4beta[1]*A[i] + gTriCub4alpha[1]*B[i] + gTriCub4gamma[1]*C[i];
		Q[i+9] = gTriCub4gamma[1]*A[i] + gTriCub4beta[1]*B[i] + gTriCub4alpha[1]*C[i];
	}

	return;
}

void KElectrostaticCubatureTriangleIntegrator::GaussPoints_Tri7P( const double* data, double* Q ) const
{
	// Calculates the 7 Gaussian points Q[0],...,Q[6]  for the 7-point cubature of the triangle
	//  A=T[0], B=T[1], C=T[2]:  corner points of the triangle
	// alpha, beta, gamma: barycentric (area) coordinates of the Gaussian points;
	// alpha+beta+gamma=1;
	// Gaussian point is weighted average of the corner points A, B, C with the alpha, beta ,gamma weights
	// See: Engeln-Müllges, Niederdrenck, Wodicka, p. 654;  Stroud 1971 (book), p. 314;
	//    Hammer, Marlowe, Stroud 1956, p. 136; Radon 1948, p. 295

	const double A[3] = { data[2], data[3], data[4] };
	const double B[3] = { data[2] + (data[0]*data[5]),
			data[3] + (data[0]*data[6]),
			data[4] + (data[0]*data[7]) };
	const double C[3] = { data[2] + (data[1]*data[8]),
			data[3] + (data[1]*data[9]),
			data[4] + (data[1]*data[10]) };

	// loop over components
	for( unsigned short i=0; i<3; i++ ) {
		Q[i] = gTriCub7alpha[0]*A[i] + gTriCub7beta[0]*B[i] + gTriCub7gamma[0]*C[i]; /* alpha A, beta B, gamma C - 0 */

		Q[i+3] = gTriCub7alpha[1]*A[i] + gTriCub7beta[1]*B[i] + gTriCub7gamma[1]*C[i]; /* alpha A, beta B, gamma C - 1 */
		Q[i+6] = gTriCub7beta[1]*A[i] + gTriCub7alpha[1]*B[i] + gTriCub7gamma[1]*C[i]; /* beta A, alpha B, gamma C - 1 */
		Q[i+9]  = gTriCub7gamma[1]*A[i] + gTriCub7beta[1]*B[i] + gTriCub7alpha[1]*C[i]; /* gamma A, beta B, alpha C - 1 */

		Q[i+12] = gTriCub7alpha[2]*A[i] + gTriCub7beta[2]*B[i] + gTriCub7gamma[2]*C[i]; /* alpha A, beta B, gamma C - 2 */
		Q[i+15] = gTriCub7beta[2]*A[i] + gTriCub7alpha[2]*B[i] + gTriCub7gamma[2]*C[i]; /* beta A, alpha B, gamma C - 2 */
		Q[i+18] = gTriCub7gamma[2]*A[i] + gTriCub7beta[2]*B[i] + gTriCub7alpha[2]*C[i]; /* gamma A, beta B, alpha C - 2 */
	}
	return;
}

void KElectrostaticCubatureTriangleIntegrator::GaussPoints_Tri12P( const double* data, double* Q ) const
{
	// Calculates the 12 Gaussian points Q[0],...,Q[11] for the 12-point cubature of the triangle (degree 7)
	// See: K. Gatermann, Computing 40, 229 (1988)
	//  A=T[0], B=T[1], C=T[2]:  corner points of the triangle
	// Area: area of the triangle calculated by Heron's formula
	// alpha, beta, gamma: barycentric (area) coordinates of the Gaussian points;
	// Gaussian point is weighted average of the corner points A, B, C with the alpha, beta ,gamma weights

	const double A[3] = { data[2], data[3], data[4] };
	const double B[3] = { data[2] + (data[0]*data[5]),
			data[3] + (data[0]*data[6]),
			data[4] + (data[0]*data[7]) };
	const double C[3] = { data[2] + (data[1]*data[8]),
			data[3] + (data[1]*data[9]),
			data[4] + (data[1]*data[10]) };

	// loop over components
	for( unsigned short i=0; i<3; i++ ) {
		Q[i] = gTriCub12alpha[0]*A[i] + gTriCub12beta[0]*B[i] + gTriCub12gamma[0]*C[i]; /* alpha A, beta B, gamma C - 0 */
		Q[i+3] = gTriCub12beta[0]*A[i] + gTriCub12gamma[0]*B[i] + gTriCub12alpha[0]*C[i]; /* beta A, gamma B, alpha C - 0 */
		Q[i+6] = gTriCub12gamma[0]*A[i] + gTriCub12alpha[0]*B[i] + gTriCub12beta[0]*C[i]; /* gamma A, alpha B, beta C - 0 */

		Q[i+9] = gTriCub12alpha[1]*A[i] + gTriCub12beta[1]*B[i] + gTriCub12gamma[1]*C[i]; /* alpha A, beta B, gamma C - 1 */
		Q[i+12] = gTriCub12beta[1]*A[i] + gTriCub12gamma[1]*B[i] + gTriCub12alpha[1]*C[i]; /* beta A, gamma B, alpha C - 1 */
		Q[i+15] = gTriCub12gamma[1]*A[i] + gTriCub12alpha[1]*B[i] + gTriCub12beta[1]*C[i]; /* gamma A, alpha B, beta C - 1 */

		Q[i+18] = gTriCub12alpha[2]*A[i] + gTriCub12beta[2]*B[i] + gTriCub12gamma[2]*C[i]; /* alpha A, beta B, gamma C - 2 */
		Q[i+21] = gTriCub12beta[2]*A[i] + gTriCub12gamma[2]*B[i] + gTriCub12alpha[2]*C[i]; /* beta A, gamma B, alpha C - 2 */
		Q[i+24] = gTriCub12gamma[2]*A[i] + gTriCub12alpha[2]*B[i] + gTriCub12beta[2]*C[i]; /* gamma A, alpha B, beta C - 2 */

		Q[i+27] = gTriCub12alpha[3]*A[i] + gTriCub12beta[3]*B[i] + gTriCub12gamma[3]*C[i]; /* alpha A, beta B, gamma C - 3 */
		Q[i+30] = gTriCub12beta[3]*A[i] + gTriCub12gamma[3]*B[i] + gTriCub12alpha[3]*C[i]; /* beta A, gamma B, alpha C - 3 */
		Q[i+33] = gTriCub12gamma[3]*A[i] + gTriCub12alpha[3]*B[i] + gTriCub12beta[3]*C[i]; /* gamma A, alpha B, beta C - 3 */
	}

	return;
}

void KElectrostaticCubatureTriangleIntegrator::GaussPoints_Tri16P( const double* data, double* Q ) const
{
	// Calculates the 16 Gaussian points Q[0],...,Q[15]  for the 16-point cubature of the triangle
	//  A=T[0], B=T[1], C=T[2]:  corner points of the triangle
	// alpha, beta, gamma: barycentric (area) coordinates of the Gaussian points;
	// Gaussian point is weighted average of the corner points A, B, C with the alpha, beta ,gamma weights

	const double A[3] = { data[2], data[3], data[4] };
	const double B[3] = { data[2] + (data[0]*data[5]),
			data[3] + (data[0]*data[6]),
			data[4] + (data[0]*data[7]) };
	const double C[3] = { data[2] + (data[1]*data[8]),
			data[3] + (data[1]*data[9]),
			data[4] + (data[1]*data[10]) };

	// loop over components
	for( unsigned short i=0; i<3; i++ ) {
		Q[i] = gTriCub16alpha[0]*A[i] + gTriCub16beta[0]*B[i] + gTriCub16gamma[0]*C[i]; /* alpha A, beta B, gamma C - 0 */

		Q[i+3] = gTriCub16alpha[1]*A[i] + gTriCub16beta[1]*B[i] + gTriCub16gamma[1]*C[i]; /* alpha A, beta B, gamma C - 1 */
		Q[i+6] = gTriCub16beta[1]*A[i] + gTriCub16alpha[1]*B[i] + gTriCub16gamma[1]*C[i]; /* beta A, alpha B, gamma C - 1 */
		Q[i+9] = gTriCub16gamma[1]*A[i] + gTriCub16beta[1]*B[i] + gTriCub16alpha[1]*C[i]; /* gamma A, beta B alpha C - 1 */

		Q[i+12] = gTriCub16alpha[2]*A[i] + gTriCub16beta[2]*B[i] + gTriCub16gamma[2]*C[i]; /* alpha A, beta B, gamma C - 2 */
		Q[i+15] = gTriCub16beta[2]*A[i] + gTriCub16alpha[2]*B[i] + gTriCub16gamma[2]*C[i]; /* beta A, alpha B, gamma C - 2 */
		Q[i+18] = gTriCub16gamma[2]*A[i] + gTriCub16beta[2]*B[i] + gTriCub16alpha[2]*C[i]; /* gamma A, beta B, alpha C - 2 */

		Q[i+21] = gTriCub16alpha[3]*A[i] + gTriCub16beta[3]*B[i] + gTriCub16gamma[3]*C[i]; /* alpha A, beta B, gamma C - 3 */
		Q[i+24] = gTriCub16beta[3]*A[i] + gTriCub16alpha[3]*B[i] + gTriCub16gamma[3]*C[i]; /* beta A, alpha B, gamma C - 3 */
		Q[i+27] = gTriCub16gamma[3]*A[i] + gTriCub16beta[3]*B[i] + gTriCub16alpha[3]*C[i]; /* gamma A, beta B, alpha C - 3 */

		Q[i+30] = gTriCub16alpha[4]*A[i] + gTriCub16beta[4]*B[i] + gTriCub16gamma[4]*C[i]; /* alpha A, beta B, gamma C - 4 */
		Q[i+33] = gTriCub16beta[4]*A[i] + gTriCub16alpha[4]*B[i] + gTriCub16gamma[4]*C[i]; /* beta A, alpha B, gamma C - 4 */
		Q[i+36] = gTriCub16gamma[4]*A[i] + gTriCub16beta[4]*B[i] + gTriCub16alpha[4]*C[i]; /* gamma A, beta B, alpha C - 4 */
		Q[i+39] = gTriCub16alpha[4]*A[i] + gTriCub16gamma[4]*B[i] + gTriCub16beta[4]*C[i]; /* alpha A, gamma B, beta C - 4 */
		Q[i+42] = gTriCub16gamma[4]*A[i] + gTriCub16alpha[4]*B[i] + gTriCub16beta[4]*C[i]; /* gamma A, alpha B, beta C - 4 */
		Q[i+45] = gTriCub16beta[4]*A[i] + gTriCub16gamma[4]*B[i] + gTriCub16alpha[4]*C[i]; /* beta A, gamma B, alpha C - 4 */
	}
}

void KElectrostaticCubatureTriangleIntegrator::GaussPoints_Tri19P( const double* data, double* Q ) const
{
	// Calculates the 19 Gaussian points Q[0],...,Q[18]  for the 19-point cubature of the triangle (degree 9).
	// See: Lyness, Jespersen, J. Inst. Math. Applics 15 (1975) 19.
	//  A=T[0], B=T[1], C=T[2]:  corner points of the triangle
	// alpha, beta, gamma: barycentric (area) coordinates of the Gaussian points;
	// Gaussian point is weighted average of the corner points A, B, C with the alpha, beta ,gamma weights

	const double A[3] = { data[2], data[3], data[4] };
	const double B[3] = { data[2] + (data[0]*data[5]),
			data[3] + (data[0]*data[6]),
			data[4] + (data[0]*data[7]) };
	const double C[3] = { data[2] + (data[1]*data[8]),
			data[3] + (data[1]*data[9]),
			data[4] + (data[1]*data[10]) };

	// loop over components
	for( unsigned short i=0; i<3; i++ ) {
		Q[i] = gTriCub19alpha[0]*A[i] + gTriCub19beta[0]*B[i] + gTriCub19gamma[0]*C[i];

		Q[i+3] = gTriCub19alpha[1]*A[i] + gTriCub19beta[1]*B[i] + gTriCub19gamma[1]*C[i];
		Q[i+6] = gTriCub19beta[1]*A[i] + gTriCub19alpha[1]*B[i] + gTriCub19gamma[1]*C[i];
		Q[i+9] = gTriCub19gamma[1]*A[i] + gTriCub19beta[1]*B[i] + gTriCub19alpha[1]*C[i];

		Q[i+12] = gTriCub19alpha[2]*A[i] + gTriCub19beta[2]*B[i] + gTriCub19gamma[2]*C[i];
		Q[i+15] = gTriCub19beta[2]*A[i] + gTriCub19alpha[2]*B[i] + gTriCub19gamma[2]*C[i];
		Q[i+18] = gTriCub19gamma[2]*A[i] + gTriCub19beta[2]*B[i] + gTriCub19alpha[2]*C[i];

		Q[i+21] = gTriCub19alpha[3]*A[i] + gTriCub19beta[3]*B[i] + gTriCub19gamma[3]*C[i];
		Q[i+24] = gTriCub19beta[3]*A[i] + gTriCub19alpha[3]*B[i] + gTriCub19gamma[3]*C[i];
		Q[i+27] = gTriCub19gamma[3]*A[i] + gTriCub19beta[3]*B[i] + gTriCub19alpha[3]*C[i];

		Q[i+30] = gTriCub19alpha[4]*A[i] + gTriCub19beta[4]*B[i] + gTriCub19gamma[4]*C[i];
		Q[i+33] = gTriCub19beta[4]*A[i] + gTriCub19alpha[4]*B[i] + gTriCub19gamma[4]*C[i];
		Q[i+36] = gTriCub19gamma[4]*A[i] + gTriCub19beta[4]*B[i] + gTriCub19alpha[4]*C[i];

		Q[i+39] = gTriCub19alpha[5]*A[i] + gTriCub19beta[5]*B[i] + gTriCub19gamma[5]*C[i];
		Q[i+42] = gTriCub19beta[5]*A[i] + gTriCub19alpha[5]*B[i] + gTriCub19gamma[5]*C[i];
		Q[i+45] = gTriCub19gamma[5]*A[i] + gTriCub19beta[5]*B[i] + gTriCub19alpha[5]*C[i];
		Q[i+48] = gTriCub19alpha[5]*A[i] + gTriCub19gamma[5]*B[i] + gTriCub19beta[5]*C[i];
		Q[i+51] = gTriCub19gamma[5]*A[i] + gTriCub19alpha[5]*B[i] + gTriCub19beta[5]*C[i];
		Q[i+54] = gTriCub19beta[5]*A[i] + gTriCub19gamma[5]*B[i] + gTriCub19alpha[5]*C[i];
	}

	return;
}

void KElectrostaticCubatureTriangleIntegrator::GaussPoints_Tri33P( const double* data, double* Q ) const
{
	// Calculates the 33 Gaussian points Q[0],...,Q[32]  for the 33-point cubature of the triangle
	// See: Papanicolopulos 2011
	//  A=T[0], B=T[1], C=T[2]:  corner points of the triangle
	// alpha, beta, gamma: barycentric (area) coordinates of the Gaussian points;
	// Gaussian point is weighted average of the corner points A, B, C with the alpha, beta ,gamma weights

	const double A[3] = { data[2], data[3], data[4] };
	const double B[3] = { data[2] + (data[0]*data[5]),
			data[3] + (data[0]*data[6]),
			data[4] + (data[0]*data[7]) };
	const double C[3] = { data[2] + (data[1]*data[8]),
			data[3] + (data[1]*data[9]),
			data[4] + (data[1]*data[10]) };

	// loop over components
	for( unsigned short i=0; i<3; i++ )
	{
		Q[i]=gTriCub33alpha[0]*A[i]+gTriCub33beta[0]*B[i]+gTriCub33gamma[0]*C[i];
		Q[i+3]=gTriCub33beta[0]*A[i]+gTriCub33alpha[0]*B[i]+gTriCub33gamma[0]*C[i];
		Q[i+6]=gTriCub33gamma[0]*A[i]+gTriCub33beta[0]*B[i]+gTriCub33alpha[0]*C[i];

		Q[i+9]=gTriCub33alpha[1]*A[i]+gTriCub33beta[1]*B[i]+gTriCub33gamma[1]*C[i];
		Q[i+12]=gTriCub33beta[1]*A[i]+gTriCub33alpha[1]*B[i]+gTriCub33gamma[1]*C[i];
		Q[i+15]=gTriCub33gamma[1]*A[i]+gTriCub33beta[1]*B[i]+gTriCub33alpha[1]*C[i];

		Q[i+18]=gTriCub33alpha[2]*A[i]+gTriCub33beta[2]*B[i]+gTriCub33gamma[2]*C[i];
		Q[i+21]=gTriCub33beta[2]*A[i]+gTriCub33alpha[2]*B[i]+gTriCub33gamma[2]*C[i];
		Q[i+24]=gTriCub33gamma[2]*A[i]+gTriCub33beta[2]*B[i]+gTriCub33alpha[2]*C[i];

		Q[i+27]=gTriCub33alpha[3]*A[i]+gTriCub33beta[3]*B[i]+gTriCub33gamma[3]*C[i];
		Q[i+30]=gTriCub33beta[3]*A[i]+gTriCub33alpha[3]*B[i]+gTriCub33gamma[3]*C[i];
		Q[i+33]=gTriCub33gamma[3]*A[i]+gTriCub33beta[3]*B[i]+gTriCub33alpha[3]*C[i];

		Q[i+36]=gTriCub33alpha[4]*A[i]+gTriCub33beta[4]*B[i]+gTriCub33gamma[4]*C[i];
		Q[i+39]=gTriCub33beta[4]*A[i]+gTriCub33alpha[4]*B[i]+gTriCub33gamma[4]*C[i];
		Q[i+42]=gTriCub33gamma[4]*A[i]+gTriCub33beta[4]*B[i]+gTriCub33alpha[4]*C[i];

		Q[i+45]=gTriCub33alpha[5]*A[i]+gTriCub33beta[5]*B[i]+gTriCub33gamma[5]*C[i];
		Q[i+48]=gTriCub33beta[5]*A[i]+gTriCub33alpha[5]*B[i]+gTriCub33gamma[5]*C[i];
		Q[i+51]=gTriCub33gamma[5]*A[i]+gTriCub33beta[5]*B[i]+gTriCub33alpha[5]*C[i];
		Q[i+54]=gTriCub33alpha[5]*A[i]+gTriCub33gamma[5]*B[i]+gTriCub33beta[5]*C[i];
		Q[i+57]=gTriCub33gamma[5]*A[i]+gTriCub33alpha[5]*B[i]+gTriCub33beta[5]*C[i];
		Q[i+60]=gTriCub33beta[5]*A[i]+gTriCub33gamma[5]*B[i]+gTriCub33alpha[5]*C[i];

		Q[i+63]=gTriCub33alpha[6]*A[i]+gTriCub33beta[6]*B[i]+gTriCub33gamma[6]*C[i];
		Q[i+66]=gTriCub33beta[6]*A[i]+gTriCub33alpha[6]*B[i]+gTriCub33gamma[6]*C[i];
		Q[i+69]=gTriCub33gamma[6]*A[i]+gTriCub33beta[6]*B[i]+gTriCub33alpha[6]*C[i];
		Q[i+72]=gTriCub33alpha[6]*A[i]+gTriCub33gamma[6]*B[i]+gTriCub33beta[6]*C[i];
		Q[i+75]=gTriCub33gamma[6]*A[i]+gTriCub33alpha[6]*B[i]+gTriCub33beta[6]*C[i];
		Q[i+78]=gTriCub33beta[6]*A[i]+gTriCub33gamma[6]*B[i]+gTriCub33alpha[6]*C[i];

		Q[i+81]=gTriCub33alpha[7]*A[i]+gTriCub33beta[7]*B[i]+gTriCub33gamma[7]*C[i];
		Q[i+84]=gTriCub33beta[7]*A[i]+gTriCub33alpha[7]*B[i]+gTriCub33gamma[7]*C[i];
		Q[i+87]=gTriCub33gamma[7]*A[i]+gTriCub33beta[7]*B[i]+gTriCub33alpha[7]*C[i];
		Q[i+90]=gTriCub33alpha[7]*A[i]+gTriCub33gamma[7]*B[i]+gTriCub33beta[7]*C[i];
		Q[i+93]=gTriCub33gamma[7]*A[i]+gTriCub33alpha[7]*B[i]+gTriCub33beta[7]*C[i];
		Q[i+96]=gTriCub33beta[7]*A[i]+gTriCub33gamma[7]*B[i]+gTriCub33alpha[7]*C[i];
	}

	return;
}

double KElectrostaticCubatureTriangleIntegrator::Potential_TriNP( const double* data, const KPosition& P, const unsigned short noPoints, double* Q, const double* weights ) const
{
	// triangle area as defined in class KTriangle: A = 0.5*fA*fB*fN1.Cross(fN2).Magnitude()
	const double area = 0.5*data[0]*data[1]*sqrt(
			POW2((data[6]*data[10]) - (data[7]*data[9])) +
			POW2((data[7]*data[8]) - (data[5]*data[10])) +
			POW2((data[5]*data[9]) - (data[6]*data[8]))
	);
	const double prefac = area * KEMConstants::OneOverFourPiEps0;

	double partialValue( 0. );
	double distVector[3] = {0., 0., 0.};
	double finalSum( 0. );
	double oneOverAbsoluteValue( 0. );
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

KEMThreeVector KElectrostaticCubatureTriangleIntegrator::ElectricField_TriNP( const double* data, const KPosition& P, const unsigned short noPoints, double* Q, const double* weights ) const
{
	// triangle area as defined in class KTriangle: A = 0.5*fA*fB*fN1.Cross(fN2).Magnitude()
	const double area = 0.5*data[0]*data[1]*sqrt(
			POW2((data[6]*data[10]) - (data[7]*data[9])) +
			POW2((data[7]*data[8]) - (data[5]*data[10])) +
			POW2((data[5]*data[9]) - (data[6]*data[8]))
	);
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

			// current vector component
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

std::pair<KEMThreeVector, double> KElectrostaticCubatureTriangleIntegrator::ElectricFieldAndPotential_TriNP( const double* data, const KPosition& P, const unsigned short noPoints, double* Q, const double* weights ) const
{
	// triangle area as defined in class KTriangle: A = 0.5*fA*fB*fN1.Cross(fN2).Magnitude()
	const double area = 0.5*data[0]*data[1]*sqrt(
			POW2((data[6]*data[10]) - (data[7]*data[9])) +
			POW2((data[7]*data[8]) - (data[5]*data[10])) +
			POW2((data[5]*data[9]) - (data[6]*data[8]))
	);
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

			// current vector component
			componentSquared = POW2(distVector[componentIndex]);

			// sum up to variable absoluteValueSquared
			absoluteValueSquared += componentSquared;

			// partialValue = P-Q_i, initialization with vector r
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

double KElectrostaticCubatureTriangleIntegrator::Potential(const KTriangle* source, const KPosition& P) const
{
	// save geometry info on triangle into array, same convention as OpenCL implementation

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

	// compute side length of triangle P1->P2

    const double triP1[3] = {
    		data[2] + (data[0]*data[5]),
    		data[3] + (data[0]*data[6]),
			data[4] + (data[0]*data[7]) }; // = fP0 + fN1*fA

    const double triP2[3] = {
    		data[2] + (data[1]*data[8]),
    		data[3] + (data[1]*data[9]),
			data[4] + (data[1]*data[10]) }; // = fP0 + fN2*fB

    const double triSideLengthP1P2 = sqrt( POW2(triP2[0]-triP1[0]) + POW2(triP2[1]-triP1[1]) + POW2(triP2[2]-triP1[2]) );

    const double size = (data[0]+data[1]+triSideLengthP1P2)/3.;

	// get distance vector from field point to centroid of triangle

    const double cen[3] = {
    		data[2] + (data[0]*data[5] + data[1]*data[8])/3.,
			data[3] + (data[0]*data[6] + data[1]*data[9])/3.,
			data[4] + (data[0]*data[7] + data[1]*data[10])/3.};

	const double dist[3] = { cen[0]-P[0], cen[1]-P[1], cen[2]-P[2] };

	// magnitude of distance vector
	const double mag = sqrt(POW2(dist[0]) + POW2(dist[1]) + POW2(dist[2]));

	// determine distance ratio (distance of field point to centroid over average triangle side length)
	const double tmpSize = 1./size;
	const double distanceRatio = mag*tmpSize;

	if( distanceRatio > fDrCutOffCub12 ) {
		// compute Gaussian points
		double triQ7[21];
		GaussPoints_Tri7P(data,triQ7);

		return Potential_TriNP( data, P, 7, triQ7, gTriCub7w );
	}
    if( distanceRatio > fDrCutOffCub33 ) {
        // compute Gaussian points
        double triQ12[36];
        GaussPoints_Tri12P(data,triQ12);

        return Potential_TriNP( data, P, 12, triQ12, gTriCub12w );
    }
	if( distanceRatio > fDrCutOffRWG ) {
		// compute Gaussian points
		double triQ33[99];
		GaussPoints_Tri33P(data,triQ33);

		return Potential_TriNP( data, P, 33, triQ33, gTriCub33w );
	}

	return KElectrostaticRWGTriangleIntegrator::Potential( source, P );
}

KEMThreeVector KElectrostaticCubatureTriangleIntegrator::ElectricField(const KTriangle* source, const KPosition& P) const
{
	// save geometry info on triangle into array, same convention as OpenCL implementation

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

	// compute side length of triangle P1->P2

    const double triP1[3] = {
    		data[2] + (data[0]*data[5]),
    		data[3] + (data[0]*data[6]),
			data[4] + (data[0]*data[7]) }; // = fP0 + fN1*fA

    const double triP2[3] = {
    		data[2] + (data[1]*data[8]),
    		data[3] + (data[1]*data[9]),
			data[4] + (data[1]*data[10]) }; // = fP0 + fN2*fB

    const double triSideLengthP1P2 = sqrt( POW2(triP2[0]-triP1[0]) + POW2(triP2[1]-triP1[1]) + POW2(triP2[2]-triP1[2]) );

    const double size = (data[0]+data[1]+triSideLengthP1P2)/3.;

	// get distance vector from field point to centroid of triangle

    const double cen[3] = {
    		data[2] + (data[0]*data[5] + data[1]*data[8])/3.,
			data[3] + (data[0]*data[6] + data[1]*data[9])/3.,
			data[4] + (data[0]*data[7] + data[1]*data[10])/3.};

	const double dist[3] = { cen[0]-P[0], cen[1]-P[1], cen[2]-P[2] };

	// magnitude of distance vector
	const double mag = sqrt(POW2(dist[0]) + POW2(dist[1]) + POW2(dist[2]));

	// determine distance ratio (distance of field point to centroid over average triangle side length)
	const double tmpSize = 1./size;
	const double distanceRatio = mag*tmpSize;

    if( distanceRatio > fDrCutOffCub12 ) {
        // compute Gaussian points
        double triQ7[21];
        GaussPoints_Tri7P(data,triQ7);

        return ElectricField_TriNP( data, P, 7, triQ7, gTriCub7w );
    }
    if( distanceRatio > fDrCutOffCub33 ) {
        // compute Gaussian points
        double triQ12[36];
        GaussPoints_Tri12P(data,triQ12);

        return ElectricField_TriNP( data, P, 12, triQ12, gTriCub12w );
    }
    if( distanceRatio > fDrCutOffRWG ) {
        // compute Gaussian points
        double triQ33[99];
        GaussPoints_Tri33P(data,triQ33);

        return ElectricField_TriNP( data, P, 33, triQ33, gTriCub33w );
    }

	return KElectrostaticRWGTriangleIntegrator::ElectricField( source, P );
}

std::pair<KEMThreeVector, double> KElectrostaticCubatureTriangleIntegrator::ElectricFieldAndPotential(const KTriangle* source, const KPosition& P) const
{
	// save geometry info on triangle into array, same convention as OpenCL implementation

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

	// compute side length of triangle P1->P2

    const double triP1[3] = {
    		data[2] + (data[0]*data[5]),
    		data[3] + (data[0]*data[6]),
			data[4] + (data[0]*data[7]) }; // = fP0 + fN1*fA

    const double triP2[3] = {
    		data[2] + (data[1]*data[8]),
    		data[3] + (data[1]*data[9]),
			data[4] + (data[1]*data[10]) }; // = fP0 + fN2*fB

    const double triSideLengthP1P2 = sqrt( POW2(triP2[0]-triP1[0]) + POW2(triP2[1]-triP1[1]) + POW2(triP2[2]-triP1[2]) );

    const double size = (data[0]+data[1]+triSideLengthP1P2)/3.;

	// get distance vector from field point to centroid of triangle

    const double cen[3] = {
    		data[2] + (data[0]*data[5] + data[1]*data[8])/3.,
			data[3] + (data[0]*data[6] + data[1]*data[9])/3.,
			data[4] + (data[0]*data[7] + data[1]*data[10])/3.};

	const double dist[3] = { cen[0]-P[0], cen[1]-P[1], cen[2]-P[2] };

	// magnitude of distance vector
	const double mag = sqrt(POW2(dist[0]) + POW2(dist[1]) + POW2(dist[2]));

	// determine distance ratio (distance of field point to centroid over average triangle side length)
	const double tmpSize = 1./size;
	const double distanceRatio = mag*tmpSize;

    if( distanceRatio > fDrCutOffCub12 ) {
        // compute Gaussian points
        double triQ7[21];
        GaussPoints_Tri7P(data,triQ7);

        return ElectricFieldAndPotential_TriNP( data, P, 7, triQ7, gTriCub7w );
    }
    if( distanceRatio > fDrCutOffCub33 ) {
        // compute Gaussian points
        double triQ12[36];
        GaussPoints_Tri12P(data,triQ12);

        return ElectricFieldAndPotential_TriNP( data, P, 12, triQ12, gTriCub12w );
    }
    if( distanceRatio > fDrCutOffRWG ) {
        // compute Gaussian points
        double triQ33[99];
        GaussPoints_Tri33P(data,triQ33);

        return ElectricFieldAndPotential_TriNP( data, P, 33, triQ33, gTriCub33w );
    }

	return KElectrostaticRWGTriangleIntegrator::ElectricFieldAndPotential( source, P );
}

double KElectrostaticCubatureTriangleIntegrator::Potential(const KSymmetryGroup<KTriangle>* source, const KPosition& P) const
{
	double potential = 0.;
	for( KSymmetryGroup<KTriangle>::ShapeCIt it=source->begin();it!=source->end();++it )
		potential += Potential(*it,P);
	return potential;
}

KEMThreeVector KElectrostaticCubatureTriangleIntegrator::ElectricField(const KSymmetryGroup<KTriangle>* source, const KPosition& P) const
{
	KEMThreeVector electricField(0.,0.,0.);
	for( KSymmetryGroup<KTriangle>::ShapeCIt it=source->begin();it!=source->end();++it )
		electricField += ElectricField(*it,P);
	return electricField;
}

std::pair<KEMThreeVector, double> KElectrostaticCubatureTriangleIntegrator::ElectricFieldAndPotential(const KSymmetryGroup<KTriangle>* source, const KPosition& P) const
{
	std::pair<KEMThreeVector, double> fieldAndPotential;
	double potential( 0. );
	KEMThreeVector electricField( 0., 0., 0. );

	for( KSymmetryGroup<KTriangle>::ShapeCIt it=source->begin(); it!=source->end(); ++it ) {
		fieldAndPotential = ElectricFieldAndPotential( *it, P );
		electricField += fieldAndPotential.first;
		potential += fieldAndPotential.second;
	}

	return std::make_pair( electricField, potential );
}

}
