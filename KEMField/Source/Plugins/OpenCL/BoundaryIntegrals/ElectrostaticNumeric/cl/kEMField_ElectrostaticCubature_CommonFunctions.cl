#ifndef KEMFIELD_ELECTROSTATICCUBATURE_COMMONFUNCTIONS_CL
#define KEMFIELD_ELECTROSTATICCUBATURE_COMMONFUNCTIONS_CL

//______________________________________________________________________________

CL_TYPE OneOverR( const CL_TYPE* Q, const CL_TYPE* P )
{
	CL_TYPE distVector0 = ( P[0] - Q[0] );
	CL_TYPE componentSquared = POW2(distVector0);
	CL_TYPE absoluteValueSquared = componentSquared;

	CL_TYPE distVector1 = ( P[1] - Q[1] );
	componentSquared = POW2(distVector1);
	absoluteValueSquared += componentSquared;

	CL_TYPE distVector2 = ( P[2] - Q[2] );
	componentSquared = POW2(distVector2);
	absoluteValueSquared += componentSquared;

	return 1./SQRT(absoluteValueSquared);
}

//______________________________________________________________________________

CL_TYPE OneOverR_VecToArr( CL_TYPE* R, const CL_TYPE* Q, const CL_TYPE* P )
{
	R[0] = ( P[0] - Q[0] );
	CL_TYPE componentSquared = POW2(R[0]);
	CL_TYPE absoluteValueSquared = componentSquared;

	R[1] = ( P[1] - Q[1] );
	componentSquared = POW2(R[1]);
	absoluteValueSquared += componentSquared;

	R[2] = ( P[2] - Q[2] );
	componentSquared = POW2(R[2]);
	absoluteValueSquared += componentSquared;

	return 1./SQRT(absoluteValueSquared);
}

//______________________________________________________________________________



#endif /*KEMFIELD_ELECTROSTATICCUBATURE_COMMONFUNCTIONS_CL*/