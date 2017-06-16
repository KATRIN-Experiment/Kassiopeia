#include "KGaussLegendreQuadrature.hh"

#include <cmath>

namespace KEMField
{
/**
 * Dr. Ferenc Glueck's numerical integration routine (Gauss Legendre).
 *
 * @param f A pointer to an array of pointers to functions to be integrated.
 * @param a Lower limit of integration.
 * @param b Upper limit of integration.
 * @param n number of integration nodes.
 * @param sum integration result.
 */

void KGaussLegendreQuadrature::operator() (double (*f)(double), double a, double b, unsigned int n, double *sum )
{
	static const double x2[1]={0.577350269189626};
	static const double w2[1]={1.};
	static const double x3[2]={0.,0.774596669241483};
	static const double w3[2]={0.888888888888889,0.555555555555556};
	static const double x4[2]={0.339981043584856,0.861136311594053};
	static const double w4[2]={0.652145154862546,0.347854845137454};
	static const double x6[3]={0.238619186083197,0.661209386466265,0.932469514203152};
	static const double w6[3]={0.467913934572691,0.360761573048139,0.171324492379170 };
	static const double x8[4]={0.183434642495650,0.525532409916329,0.796666477413627,0.960289856497536 };
	static const double w8[4]={0.362683783378362,0.313706645877887,0.222381034453374,0.101228536290376};
	static const double x16[8]={0.09501250983763744,0.28160355077925891,0.45801677765722739,0.61787624440264375,
			0.75540440835500303,0.86563120238783174,0.94457502307323258 ,0.98940093499164993};
	static const double w16[8]={0.189450610455068496,0.182603415044923589,0.169156519395002532,0.149595988816576731,
			0.124628971255533872,0.095158511682492785,0.062253523938647892,0.027152459411754095};
	static const double x32[16]={0.048307665687738316,0.144471961582796493,0.239287362252137075 ,0.331868602282127650,
			0.421351276130635345,0.506899908932229390,0.587715757240762329,0.663044266930215201,
			0.732182118740289680,0.794483795967942407,0.849367613732569970,0.896321155766052124,
			0.934906075937739689,0.964762255587506431,0.985611511545268335,0.997263861849481564};
	static const double w32[16]={0.09654008851472780056,0.09563872007927485942,0.09384439908080456564,0.09117387869576388471,
			0.08765209300440381114,0.08331192422694675522,0.07819389578707030647,0.07234579410884850625,
			0.06582222277636184684,0.05868409347853554714,0.05099805926237617619,0.04283589802222680057,
			0.03427386291302143313,0.02539206530926205956,0.01627439473090567065,0.00701861000947009660};


	unsigned int j;
	double A,B;

	if( n<=2 )
		n=2;
	else if( n>=5 && n<=6 )
		n=6;
	else if(n>=7 && n<=8)
		n=8;
	else if(n>=9 && n<=16)
		n=16;
	else  if(n>16)
		n=32;

	A=(b-a)/2.;
	B=(b+a)/2.;
	sum[0]=0.;
	//  printf("n=%12i \t\n",n);

	if(n==2)
	{
		sum[0]=w2[0]*f(B+A*x2[0])+w2[0]*f(B-A*x2[0]);
	}
	else if(n==3)
	{
		sum[0]=w3[0]*f(B+A*x3[0])+w3[1]*f(B+A*x3[1])+w3[1]*f(B-A*x3[1]);
	}
	else if(n==4)
	{
		for(j=0;j<=1;j++)
			sum[0]+=w4[j]*f(B+A*x4[j]);
		for(j=0;j<=1;j++)
			sum[0]+=w4[j]*f(B-A*x4[j]);
	}
	else if(n==6)
	{
		for(j=0;j<=2;j++)
			sum[0]+=w6[j]*f(B+A*x6[j]);
		for(j=0;j<=2;j++)
			sum[0]+=w6[j]*f(B-A*x6[j]);
	}
	else if(n==8)
	{
		for(j=0;j<=3;j++)
			sum[0]+=w8[j]*f(B+A*x8[j]);
		for(j=0;j<=3;j++)
			sum[0]+=w8[j]*f(B-A*x8[j]);
	}
	else if(n==16)
	{
		for(j=0;j<=7;j++)
			sum[0]+=w16[j]*f(B+A*x16[j]);
		for(j=0;j<=7;j++)
			sum[0]+=w16[j]*f(B-A*x16[j]);
	}
	else if(n==32)
	{
		for(j=0;j<=15;j++)
			sum[0]+=w32[j]*f(B+A*x32[j]);
		for(j=0;j<=15;j++)
			sum[0]+=w32[j]*f(B-A*x32[j]);
	}

	sum[0]=A*sum[0];

	return;
}



} /* KEMField namespace */
