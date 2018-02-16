#include "KElectrostaticQuadratureLineSegmentIntegrator.hh"

#define POW2(x) ((x)*(x))
#define POW3(x) ((x)*(x)*(x))


namespace KEMField
{
double KElectrostaticQuadratureLineSegmentIntegrator::Potential_nNodes( const double* data, const KPosition& P,
		const unsigned short halfNoNodes, const double* nodes, const double* weights ) const
{
	const double lineCenter[3] = {
			0.5*(data[0] + data[3]),
			0.5*(data[1] + data[4]),
			0.5*(data[2] + data[5])
	};

	const double lineLength = sqrt( POW2(data[3] - data[0]) + POW2(data[4] - data[1])	+ POW2(data[5] - data[2]) );
	const double halfLength = 0.5 * lineLength;

	const double prefacUnit = 1./lineLength;
	const double lineUnit[3] = { prefacUnit * (data[3] - data[0]),
			prefacUnit * (data[4] - data[1]),
			prefacUnit * (data[5] - data[2]) };

	double posDir[3] = {0., 0., 0.};
	double posMag( 0. );
	double negDir[3] = {0., 0., 0.};
	double negMag( 0. );

	double weightFac( 0. );
	double nodeIt( 0. );

	double sum( 0. );

	// loop over nodes
	for( unsigned short i=0; i<halfNoNodes; i++ ) {
		weightFac = weights[i] * halfLength;
		nodeIt = nodes[i] * halfLength;

		// reset variables for magnitude
		posMag = 0.;
		negMag = 0.;

		// loop over components
		for (unsigned short j = 0; j < 3; j++) {
			// positive line direction
			posDir[j] = P[j] - (lineCenter[j] + (lineUnit[j] * nodeIt));
			posMag += POW2(posDir[j]);
			// negative line direction
			negDir[j] = P[j] - (lineCenter[j] - (lineUnit[j] * nodeIt));
			negMag += POW2(negDir[j]);
		}

		posMag = 1. / sqrt( posMag );
		negMag = 1. / sqrt( negMag );

		sum += weightFac * (posMag + negMag);
	}

	const double oneOverEps0 = 1./KEMConstants::Eps0;

	return (0.25*oneOverEps0*sum*data[6]);
}

KEMThreeVector KElectrostaticQuadratureLineSegmentIntegrator::ElectricField_nNodes( const double* data, const KPosition& P,
		const unsigned short halfNoNodes, const double* nodes, const double* weights ) const
{
	const double lineCenter[3] = {
			0.5*(data[0] + data[3]),
			0.5*(data[1] + data[4]),
			0.5*(data[2] + data[5])
	};

	const double lineLength = sqrt( POW2(data[3] - data[0]) + POW2(data[4] - data[1])	+ POW2(data[5] - data[2]) );
	const double halfLength = 0.5 * lineLength;

	const double prefacUnit = 1./lineLength;
	const double lineUnit[3] = { prefacUnit * (data[3] - data[0]),
			prefacUnit * (data[4] - data[1]),
			prefacUnit * (data[5] - data[2]) };

	double posDir[3] = {0., 0., 0.};
	double posMag( 0. );
	double negDir[3] = {0., 0., 0.};
	double negMag( 0. );

	double weightFac( 0. );
	double nodeIt( 0. );

	double sum[3] = { 0., 0., 0. };

	// loop over nodes
	for( unsigned short i=0; i<halfNoNodes; i++ ) {
		weightFac = weights[i] * halfLength;
		nodeIt = nodes[i] * halfLength;

		// reset variables for magnitude
		posMag = 0.;
		negMag = 0.;

		// loop over components
		for( unsigned short j=0; j<3; j++ ) {
			// positive line direction
			posDir[j] = P[j] - ( lineCenter[j] + (lineUnit[j]*nodeIt) );
			posMag += POW2(posDir[j]);
			// negative line direction
			negDir[j] = P[j] - ( lineCenter[j] - (lineUnit[j]*nodeIt) );
			negMag += POW2(negDir[j]);
		}

		posMag = 1. / sqrt( posMag );
		posMag = POW3( posMag );
		negMag = 1. / sqrt( negMag );
		negMag = POW3( negMag );

		for( unsigned short k=0; k<3; k++ ) {
			sum[k] += weightFac * (posMag*posDir[k] + negMag*negDir[k]);
		}
	}

	const double prefac = 0.25*data[6]/KEMConstants::Eps0;
	double EField[3];
	for( unsigned short l=0; l<3; l++ ) {
		EField[l] = prefac * sum[l];
	}

	return KEMThreeVector( EField[0], EField[1], EField[2] );
}

std::pair<KEMThreeVector,double> KElectrostaticQuadratureLineSegmentIntegrator::ElectricFieldAndPotential_nNodes( const double* data, const KPosition& P,
		const unsigned short halfNoNodes, const double* nodes, const double* weights ) const
{
	const double lineCenter[3] = {
			0.5*(data[0] + data[3]),
			0.5*(data[1] + data[4]),
			0.5*(data[2] + data[5])
	};

	const double lineLength = sqrt( POW2(data[3] - data[0]) + POW2(data[4] - data[1])	+ POW2(data[5] - data[2]) );
	const double halfLength = 0.5 * lineLength;

	const double prefacUnit = 1./lineLength;
	const double lineUnit[3] = { prefacUnit * (data[3] - data[0]),
			prefacUnit * (data[4] - data[1]),
			prefacUnit * (data[5] - data[2]) };

	double posDir[3] = {0., 0., 0.};
	double posMag( 0. );
	double negDir[3] = {0., 0., 0.};
	double negMag( 0. );

	double weightFac( 0. );
	double nodeIt( 0. );

	double sumField[3] = { 0., 0., 0. };
	double sumPhi = 0.;

	// loop over nodes
	for( unsigned short i=0; i<halfNoNodes; i++ ) {
		weightFac = weights[i] * halfLength;
		nodeIt = nodes[i] * halfLength;

		// reset variables for magnitude
		posMag = 0.;
		negMag = 0.;

		// loop over components
		for( unsigned short j=0; j<3; j++ ) {
			// positive line direction
			posDir[j] = P[j] - ( lineCenter[j] + (lineUnit[j]*nodeIt) );
			posMag += POW2(posDir[j]);
			// negative line direction
			negDir[j] = P[j] - ( lineCenter[j] - (lineUnit[j]*nodeIt) );
			negMag += POW2(negDir[j]);
		}

		posMag = 1. / sqrt( posMag );
		negMag = 1. / sqrt( negMag );

		sumPhi += weightFac * (posMag + negMag);

		posMag = POW3( posMag );
		negMag = POW3( negMag );

		for( unsigned short k=0; k<3; k++ ) {
			sumField[k] += weightFac * (posMag*posDir[k] + negMag*negDir[k]);
		}
	}

	const double prefac = 0.25*data[6]/KEMConstants::Eps0;
	double EField[3];
	double Phi;
	for( unsigned short l=0; l<3; l++ ) {
		EField[l] = prefac * sumField[l];
	}
	Phi = prefac * sumPhi;
	return std::make_pair( KEMThreeVector(EField[0], EField[1], EField[2]), Phi );
}

double KElectrostaticQuadratureLineSegmentIntegrator::Potential(const KLineSegment* source,const KPosition& P) const
{
	// save geometry info on triangle into array, same convention as OpenCL implementation

	const double data[7] = {
			source->GetP0().X(),
			source->GetP0().Y(),
			source->GetP0().Z(),
			source->GetP1().X(),
			source->GetP1().Y(),
			source->GetP1().Z(),
			source->GetDiameter() };

    const double length = sqrt(
    		((data[3]-data[0])*(data[3]-data[0]))
    		+ ((data[4]-data[1])*(data[4]-data[1]))
			+ ((data[5]-data[2])*(data[5]-data[2])));

	// get distance vector from field point to centroid of line segment

    const double cen[3] = {
    		(data[0] + data[3])*0.5,
			(data[1] + data[4])*0.5,
			(data[2] + data[5])*0.5};

	const double dist[3] = { cen[0]-P[0], cen[1]-P[1], cen[2]-P[2] };

	// magnitude of distance vector
	const double mag = sqrt(POW2(dist[0]) + POW2(dist[1]) + POW2(dist[2]));

	// determine distance ratio (distance of field point to centroid over line segment length)
	const double tmpSize = 1./length;
	const double distanceRatio = mag*tmpSize;

	if( distanceRatio > fDrCutOffQuad16 ) {
		return Potential_nNodes( data, P, 2, gQuadx4, gQuadw4 );
	}
	if( distanceRatio > fDrCutOffAna ) {
		return Potential_nNodes( data, P, 8, gQuadx16, gQuadw16 );
	}

	return KElectrostaticAnalyticLineSegmentIntegrator::Potential( source, P );
}


KEMThreeVector KElectrostaticQuadratureLineSegmentIntegrator::ElectricField(const KLineSegment* source,const KPosition& P) const
{
	// save geometry info on triangle into array, same convention as OpenCL implementation

	const double data[7] = {
			source->GetP0().X(),
			source->GetP0().Y(),
			source->GetP0().Z(),
			source->GetP1().X(),
			source->GetP1().Y(),
			source->GetP1().Z(),
			source->GetDiameter() };

    const double length = sqrt(
    		((data[3]-data[0])*(data[3]-data[0]))
    		+ ((data[4]-data[1])*(data[4]-data[1]))
			+ ((data[5]-data[2])*(data[5]-data[2])));

	// get distance vector from field point to centroid of line segment

    const double cen[3] = {
    		(data[0] + data[3])*0.5,
			(data[1] + data[4])*0.5,
			(data[2] + data[5])*0.5};

	const double dist[3] = { cen[0]-P[0], cen[1]-P[1], cen[2]-P[2] };

	// magnitude of distance vector
	const double mag = sqrt(POW2(dist[0]) + POW2(dist[1]) + POW2(dist[2]));

	// determine distance ratio (distance of field point to centroid over line segment length)
	const double tmpSize = 1./length;
	const double distanceRatio = mag*tmpSize;

	if( distanceRatio > fDrCutOffQuad16 ) {
		return ElectricField_nNodes( data, P, 2, gQuadx4, gQuadw4 );
	}
	if( distanceRatio > fDrCutOffAna ) {
		return ElectricField_nNodes( data, P, 8, gQuadx16, gQuadw16 );
	}

	return KElectrostaticAnalyticLineSegmentIntegrator::ElectricField( source, P );
}

std::pair<KEMThreeVector, double> KElectrostaticQuadratureLineSegmentIntegrator::ElectricFieldAndPotential(const KLineSegment* source, const KPosition& P) const
{
	// save geometry info on triangle into array, same convention as OpenCL implementation

	const double data[7] = {
			source->GetP0().X(),
			source->GetP0().Y(),
			source->GetP0().Z(),
			source->GetP1().X(),
			source->GetP1().Y(),
			source->GetP1().Z(),
			source->GetDiameter() };

    const double length = sqrt(
    		((data[3]-data[0])*(data[3]-data[0]))
    		+ ((data[4]-data[1])*(data[4]-data[1]))
			+ ((data[5]-data[2])*(data[5]-data[2])));

	// get distance vector from field point to centroid of line segment

    const double cen[3] = {
    		(data[0] + data[3])*0.5,
			(data[1] + data[4])*0.5,
			(data[2] + data[5])*0.5};

	const double dist[3] = { cen[0]-P[0], cen[1]-P[1], cen[2]-P[2] };

	// magnitude of distance vector
	const double mag = sqrt(POW2(dist[0]) + POW2(dist[1]) + POW2(dist[2]));

	// determine distance ratio (distance of field point to centroid over line segment length)
	const double tmpSize = 1./length;
	const double distanceRatio = mag*tmpSize;

	if( distanceRatio > fDrCutOffQuad16 ) {
		return ElectricFieldAndPotential_nNodes( data, P, 2, gQuadx4, gQuadw4 );
	}
	if( distanceRatio > fDrCutOffAna ) {
		return ElectricFieldAndPotential_nNodes( data, P, 8, gQuadx16, gQuadw16 );
	}

	return KElectrostaticAnalyticLineSegmentIntegrator::ElectricFieldAndPotential( source, P );
}

double KElectrostaticQuadratureLineSegmentIntegrator::Potential(const KSymmetryGroup<KLineSegment>* source, const KPosition& P) const
{
	double potential = 0.;
	for (KSymmetryGroup<KLineSegment>::ShapeCIt it=source->begin();it!=source->end();++it)
		potential += Potential(*it,P);
	return potential;
}

KEMThreeVector KElectrostaticQuadratureLineSegmentIntegrator::ElectricField(const KSymmetryGroup<KLineSegment>* source, const KPosition& P) const
{
	KEMThreeVector electricField(0.,0.,0.);
	for (KSymmetryGroup<KLineSegment>::ShapeCIt it=source->begin();it!=source->end();++it)
		electricField += ElectricField(*it,P);
	return electricField;
}

std::pair<KEMThreeVector, double> KElectrostaticQuadratureLineSegmentIntegrator::ElectricFieldAndPotential( const KSymmetryGroup<KLineSegment>* source, const KPosition& P ) const
{
	std::pair<KEMThreeVector, double> fieldAndPotential;
    double potential( 0. );
    KEMThreeVector electricField( 0., 0., 0. );

    for( KSymmetryGroup<KLineSegment>::ShapeCIt it=source->begin(); it!=source->end(); ++it ) {
    	fieldAndPotential = ElectricFieldAndPotential( *it, P );
        electricField += fieldAndPotential.first;
    	potential += fieldAndPotential.second;
    }

    return std::make_pair( electricField, potential );
}

}
