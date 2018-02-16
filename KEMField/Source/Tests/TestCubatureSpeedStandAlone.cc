#include <iostream>
#include <time.h>
#include <math.h>

#define POW2(x) ((x)*(x))
#define POW3(x) ((x)*(x)*(x))

// constants needed for stand-alone integrator functions below
#define M_ONEOVER_4PI_EPS0 8987551787.9979107161559640186992

#include "include/TestRWGTriangleIntegrator.hh"
#include "include/TestCubatureTriangleIntegrator.hh"

// number of triangles
#define NUMTRI 1500000

// number of field points to be computed
#define POINTS 100

// assuming constant charge density as prefactor to be multiplied
#define CHDEN 0.01

// global data arrays on heap
double triangleData[11*NUMTRI], triangleQ7[21*NUMTRI], triangleQ12[36*NUMTRI];

// random number seed
int IJKLRANDOM;

// steps after integration
//#define SUMPOT /* perform summation for getting potential at field point */
//#define SUMFIELD /* perform summation for getting field at field point */

void subrn(double *u,int len);
double randomnumber();

clock_t start;

void StartTimer()
{
	start = clock();
}

double Time()
{
	double end = clock();
	return ((double)(end - start))/CLOCKS_PER_SEC; // time in seconds
}

void RWG( double* data, const double* P, double* res )
{
	for( unsigned int i=0; i<NUMTRI; i++ ) {
		FieldPotTriRWG( &data[11*i], P, res );
#ifdef SUMFIELD
		for( unsigned short j=0; j<3; j++ ) res[j] += ( CHDEN * res[j] );
#endif
#ifdef SUMPOT
		res[3] += ( CHDEN * res[3] );
#endif
	}
}

void Cubature7PointCached( double* data, double* Q7, const double* P, double* res )
{
	for( unsigned int i=0; i<NUMTRI; i++ ) {
		FieldPotTri7P_Cached( &data[11*i], P, &Q7[21*i], res ); // electric field and pot
#ifdef SUMFIELD
		for( unsigned short j=0; j<3; j++ ) res[j] += ( CHDEN * res[j] );
#endif
#ifdef SUMPOT
		res[3] += ( CHDEN * res[3] );
#endif
	}
}

void Cubature7PointNonCached( double* data, const double* P, double* res )
{
	for( unsigned int i=0; i<NUMTRI; i++ ) {
		FieldPotTri7P_NonCached( &data[11*i], P, res );
#ifdef SUMFIELD
		for( unsigned short j=0; j<3; j++ ) res[j] += ( CHDEN * res[j] );
#endif
#ifdef SUMPOT
		res[3] += ( CHDEN * res[3] );
#endif
	}
}

void Cubature12PointCached( double* data, double* Q12, const double* P, double* res )
{
	for( unsigned int i=0; i<NUMTRI; i++ ) {
		FieldPotTri12P_Cached( &data[11*i], P, &Q12[36*i], res );
#ifdef SUMFIELD
		for( unsigned short j=0; j<3; j++ ) res[j] += ( CHDEN * res[j] );
#endif
#ifdef SUMPOT
		res[3] += ( CHDEN * res[3] );
#endif
	}
}

void Cubature12PointNonCached( double* data, const double* P, double* res )
{
	for( unsigned int i=0; i<NUMTRI; i++ ) {
		FieldPot12P_NonCached( &data[11*i], P, res );
#ifdef SUMFIELD
		for( unsigned short j=0; j<3; j++ ) res[j] += ( CHDEN * res[j] );
#endif
#ifdef SUMPOT
		res[3] += ( CHDEN * res[3] );
#endif
	}
}

int main()
{
	// Speed test for saved Q points for triangle 7- and 12-point cubature
	// here we don't check the distance ratio of each triangle in the container

	std::cout << "Speed test, computation of field and potential simultaneously." << std::endl;

#ifdef SUMPOT
	std::cout << "Element summation for potential." << std::endl;
#endif
#ifdef SUMFIELD
	std::cout << "Element summation for field." << std::endl;
#endif
#ifndef SUMFIELD
#ifndef SUMPOT
	std::cout << "No element summation." << std::endl << std::endl;
#endif
#endif

	double costheta = 0.;
	double sintheta = 0.;
	double phi = 0.;
	double r = 100.;

	// shape data
	double P0[3] = {0., 0., 0.};
	double P1[3] = {0., 0., 0.};
	double P2[3] = {0., 0., 0.};

	double A = 0.;
	double B = 0.;
	// P0
	double N1[3] = {0., 0., 0.};
	double N2[3] = {0., 0., 0.};

	// field point and potential value
	double fP[3] = { 0., 0., 0. };

	double time7Points( 0. ), time12Points( 0. );

	for( unsigned int i=0; i<NUMTRI; i++ ) {

		IJKLRANDOM = i+1;

		// dice triangle geometry
		for( unsigned short l=0; l<3; l++ ) P0[l]=-1.+2.*randomnumber();
		for( unsigned short j=0; j<3; j++ ) P1[j]=-1.+2.*randomnumber();
		for( unsigned short k=0; k<3; k++ ) P2[k]=-1.+2.*randomnumber();

		// compute further triangle data
		A = sqrt(POW2(P1[0]-P0[0]) + POW2(P1[1]-P0[1]) + POW2(P1[2]-P0[2]));
		B = sqrt(POW2(P2[0]-P0[0]) + POW2(P2[1]-P0[1]) + POW2(P2[2]-P0[2]));

		N1[0] = (P1[0]-P0[0]) / A;
		N1[1] = (P1[1]-P0[1]) / A;
		N1[2] = (P1[2]-P0[2]) / A;
		N2[0] = (P2[0]-P0[0]) / B;
		N2[1] = (P2[1]-P0[1]) / B;
		N2[2] = (P2[2]-P0[2]) / B;

		// save triangle data into global array
		triangleData[11*i+0] = A;
		triangleData[11*i+1] = B;
		triangleData[11*i+2] = P0[0];
		triangleData[11*i+3] = P0[1];
		triangleData[11*i+4] = P0[2];
		triangleData[11*i+5] = N1[0];
		triangleData[11*i+6] = N1[1];
		triangleData[11*i+7] = N1[2];
		triangleData[11*i+8] = N2[0];
		triangleData[11*i+9] = N2[1];
		triangleData[11*i+10] = N2[2];

		// compute 7-points (21) and 12-points (36)
		StartTimer();
		GaussPoints_Tri7P( &triangleData[11*i] , &triangleQ7[21*i] );
		time7Points += Time();

		StartTimer();
		GaussPoints_Tri12P( &triangleData[11*i], &triangleQ12[36*i] );
		time12Points += Time();
	}

	std::cout << "[X] Time for computing 7-points of " << NUMTRI << " triangles: " << time7Points << std::endl;
	std::cout << "[X] Time for computing 12-points of " << NUMTRI << " triangles: " << time12Points << std::endl;

	// further time values

	double timeRWG(0.), timeCub7Cached(0.), timeCub7NonCached(0.), timeCub12Cached(0.), timeCub12NonCached(0.);

	double fieldAndPotential[4] = {0., 0., 0., 0.};

	for( unsigned int i=0; i<POINTS; i++ ) {

		// dice direction vector field point

		costheta=-1.+2.*randomnumber();
		sintheta=sqrt(1.-costheta*costheta);
		phi=2.*M_PI*randomnumber();

		fP[0]=r*sintheta*cos(phi);
		fP[1]=r*sintheta*sin(phi);
		fP[2]=r*costheta;

		// compute field and potential with RWG

		StartTimer();
		RWG( triangleData, fP, fieldAndPotential );
		timeRWG += Time();

		// compute 7-points with saved points -> measure time

		StartTimer();
		Cubature7PointCached( triangleData, triangleQ7, fP, fieldAndPotential );
		timeCub7Cached += Time();

		// compute 7-points without saved points -> measure time

		StartTimer();
		Cubature7PointNonCached( triangleData, fP, fieldAndPotential );
		timeCub7NonCached += Time();

		// compute 12-points with saved points -> measure time

		StartTimer();
		Cubature12PointCached( triangleData, triangleQ12, fP, fieldAndPotential );
		timeCub12Cached += Time();

		// compute 12-points without saved points -> measure time

		StartTimer();
		Cubature12PointNonCached( triangleData, fP, fieldAndPotential );
		timeCub12NonCached += Time();
	}

	std::cout << "Time results (s) for " << NUMTRI << " triangles and " << POINTS << " evaluation points:" << std::endl;

	std::cout << "[X] RWG              f+p: " << timeRWG << std::endl;
	std::cout << "[X] 7-point, cached  f+p: " << timeCub7Cached << std::endl;
	std::cout << "[X] 7-point          f+p: " << timeCub7NonCached << std::endl;
	std::cout << "[X] 12-point, cached f+p: " << timeCub12Cached << std::endl;
	std::cout << "[X] 12-point         f+p: " << timeCub12NonCached << std::endl << std::endl;

	std::cout << "Time ratios:" << std::endl;

	std::cout << "[X] RWG / 7-point, cached   f+p: " << timeRWG/timeCub7Cached << std::endl;
	std::cout << "[X] RWG / 7-point           f+p: " << timeRWG/timeCub7NonCached << std::endl;
	std::cout << "[X] RWG / 12-point, cached  f+p: " << timeRWG/timeCub12Cached << std::endl;
	std::cout << "[X] RWG / 12-point          f+p: " << timeRWG/timeCub12NonCached << std::endl;

	return 0;
}

void subrn(double *u,int len)
{
	// This subroutine computes random numbers u[1],...,u[len]
	// in the (0,1) interval. It uses the 0<IJKLRANDOM<900000000
	// integer as initialization seed.
	//  In the calling program the dimension
	// of the u[] vector should be larger than len (the u[0] value is
	// not used).
	// For each IJKLRANDOM
	// numbers the program computes completely independent random number
	// sequences (see: F. James, Comp. Phys. Comm. 60 (1990) 329, sec. 3.3).
	static int iff=0;
	static long ijkl,ij,kl,i,j,k,l,ii,jj,m,i97,j97,ivec;
	static float s,t,uu[98],c,cd,cm,uni;
	if(iff==0)
	{
		if(IJKLRANDOM==0)
		{
			std::cout << "Message from subroutine subrn:\n";
			std::cout << "the global integer IJKLRANDOM should be larger than 0 !!!\n";
			std::cout << "Computation is  stopped !!! \n";
		}
		ijkl=IJKLRANDOM;
		if(ijkl<1 || ijkl>=900000000) ijkl=1;
		ij=ijkl/30082;
		kl=ijkl-30082*ij;
		i=((ij/177)%177)+2;
		j=(ij%177)+2;
		k=((kl/169)%178)+1;
		l=kl%169;
		for(ii=1;ii<=97;ii++)
		{ s=0; t=0.5;
		for(jj=1;jj<=24;jj++)
		{ m=(((i*j)%179)*k)%179;
		i=j; j=k; k=m;
		l=(53*l+1)%169;
		if((l*m)%64 >= 32) s=s+t;
		t=0.5*t;
		}
		uu[ii]=s;
		}
		c=362436./16777216.;
		cd=7654321./16777216.;
		cm=16777213./16777216.;
		i97=97;
		j97=33;
		iff=1;
	}
	for(ivec=1;ivec<=len;ivec++)
	{ uni=uu[i97]-uu[j97];
	if(uni<0.) uni=uni+1.;
	uu[i97]=uni;
	i97=i97-1;
	if(i97==0) i97=97;
	j97=j97-1;
	if(j97==0) j97=97;
	c=c-cd;
	if(c<0.) c=c+cm;
	uni=uni-c;
	if(uni<0.) uni=uni+1.;
	if(uni==0.)
	{ uni=uu[j97]*0.59604644775391e-07;
	if(uni==0.) uni=0.35527136788005e-14;
	}
	u[ivec]=uni;
	}
	return;
}

double randomnumber()
{
	// This function computes 1 random number in the (0,1) interval,
	// using the subrn subroutine.

	double u[2];
	subrn(u,1);
	return u[1];
}
