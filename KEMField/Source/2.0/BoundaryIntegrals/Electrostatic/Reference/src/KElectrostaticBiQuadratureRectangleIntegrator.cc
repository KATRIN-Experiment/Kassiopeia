#include "KElectrostaticBiQuadratureRectangleIntegrator.hh"

namespace KEMField
{

// global variables

unsigned short gReField;
unsigned int gReIntNodes = 32;

double reAcommon, reBcommon;

KEMThreeVector reP, reP0, gReN1, gReN2;
double gReX;


double KElectrostaticBiQuadratureRectangleIntegrator::rectF1( double x )
{
	gReX = x;
	double ret = rectQuadGaussLegendreVarN(&KElectrostaticBiQuadratureRectangleIntegrator::rectF2,0.,reBcommon, gReIntNodes);

	return ret;
}

double KElectrostaticBiQuadratureRectangleIntegrator::rectF2( double y )
{
	  double x = gReX;
	  return KElectrostaticBiQuadratureRectangleIntegrator::rectF( x, y );
}

double KElectrostaticBiQuadratureRectangleIntegrator::rectF( double x, double y )
{
	double R,R3;
	KEMThreeVector Q,QP;
	Q = reP0 + x*gReN1 + y*gReN2;
	QP=reP-Q;
	R=QP.Magnitude();
	R3=R*R*R;

	// return electric potential
	if( gReField==3 )
		return KEMConstants::OneOverFourPiEps0/R;

	// return electric field
	for( unsigned short j=0;j<3;j++ ) {
		if( gReField==j )
			return KEMConstants::OneOverFourPiEps0/R3*QP[j];
	}


	return 0.;
}

double KElectrostaticBiQuadratureRectangleIntegrator::rectQuadGaussLegendreVarN( double (*f)(double), double a, double b, unsigned int n )
{
   KGaussLegendreQuadrature fIntegrator;
   double Integral,xmin,xmax,del, ret;
   if( n<=32 )
	   fIntegrator(f,a,b,n,&Integral);
   else
   {
      unsigned int imax=n/32+1;
      Integral=0.;
      del=(b-a)/imax;
      for( unsigned int i=1; i<=imax; i++ )
      {
    	  xmin=a+del*(i-1);
    	  xmax=xmin+del;
    	  fIntegrator(f,xmin,xmax,32, &ret);
    	  Integral+=ret;
      }
   }
  return Integral;
}

double KElectrostaticBiQuadratureRectangleIntegrator::Potential(const KRectangle* source, const KPosition& P) const
{
	gReField = 3;

	reP = P;
	reP0 = source->GetP0();
	gReN1 = source->GetN1();
	gReN2 = source->GetN2();
	reAcommon=source->GetA();
	reBcommon=source->GetB();

	return rectQuadGaussLegendreVarN( &KElectrostaticBiQuadratureRectangleIntegrator::rectF1, 0., source->GetA(), gReIntNodes );
}

KEMThreeVector KElectrostaticBiQuadratureRectangleIntegrator::ElectricField(const KRectangle* source, const KPosition& P) const
{
	double EField[3];

	reP=P;
	reP0=source->GetP0();
	gReN1=source->GetN1();
	gReN2=source->GetN2();
	reAcommon=source->GetA();
	reBcommon=source->GetB();

	for( unsigned short j=0; j<3; j++ ) {
		gReField = j;
		EField[j] = rectQuadGaussLegendreVarN( &KElectrostaticBiQuadratureRectangleIntegrator::rectF1, 0., source->GetA(), gReIntNodes );
	}

	return KEMThreeVector( EField[0], EField[1], EField[2] );
}

std::pair<KEMThreeVector, double> KElectrostaticBiQuadratureRectangleIntegrator::ElectricFieldAndPotential(const KRectangle* source, const KPosition& P) const
{
	return std::make_pair( ElectricField(source, P), Potential(source, P) );
}

double KElectrostaticBiQuadratureRectangleIntegrator::Potential(const KSymmetryGroup<KRectangle>* source, const KPosition& P) const
{
	double potential = 0.;
	for (KSymmetryGroup<KRectangle>::ShapeCIt it=source->begin();it!=source->end();++it)
		potential += Potential(*it,P);
	return potential;
}

KEMThreeVector KElectrostaticBiQuadratureRectangleIntegrator::ElectricField(const KSymmetryGroup<KRectangle>* source, const KPosition& P) const
{
	KEMThreeVector electricField(0.,0.,0.);
	for (KSymmetryGroup<KRectangle>::ShapeCIt it=source->begin();it!=source->end();++it)
		electricField += ElectricField(*it,P);
	return electricField;
}

std::pair<KEMThreeVector, double> KElectrostaticBiQuadratureRectangleIntegrator::ElectricFieldAndPotential(const KSymmetryGroup<KRectangle>* source, const KPosition& P) const
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
