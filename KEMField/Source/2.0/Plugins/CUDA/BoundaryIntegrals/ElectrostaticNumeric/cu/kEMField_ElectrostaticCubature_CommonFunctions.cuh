#ifndef KEMFIELD_ELECTROSTATICCUBATURE_COMMONFUNCTIONS_CUH
#define KEMFIELD_ELECTROSTATICCUBATURE_COMMONFUNCTIONS_CUH

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE OneOverR( const CU_TYPE* Q, const CU_TYPE* P )
{
	CU_TYPE distVector0 = ( P[0] - Q[0] );
	CU_TYPE componentSquared = POW2(distVector0);
	CU_TYPE absoluteValueSquared = componentSquared;

	CU_TYPE distVector1 = ( P[1] - Q[1] );
	componentSquared = POW2(distVector1);
	absoluteValueSquared += componentSquared;

	CU_TYPE distVector2 = ( P[2] - Q[2] );
	componentSquared = POW2(distVector2);
	absoluteValueSquared += componentSquared;

	return 1./SQRT(absoluteValueSquared);
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE OneOverR_VecToArr( CU_TYPE* R, const CU_TYPE* Q, const CU_TYPE* P )
{
	R[0] = ( P[0] - Q[0] );
	CU_TYPE componentSquared = POW2(R[0]);
	CU_TYPE absoluteValueSquared = componentSquared;

	R[1] = ( P[1] - Q[1] );
	componentSquared = POW2(R[1]);
	absoluteValueSquared += componentSquared;

	R[2] = ( P[2] - Q[2] );
	componentSquared = POW2(R[2]);
	absoluteValueSquared += componentSquared;

	return 1./SQRT(absoluteValueSquared);
}

//______________________________________________________________________________



#endif /*KEMFIELD_ELECTROSTATICCUBATURE_COMMONFUNCTIONS_CUH*/
