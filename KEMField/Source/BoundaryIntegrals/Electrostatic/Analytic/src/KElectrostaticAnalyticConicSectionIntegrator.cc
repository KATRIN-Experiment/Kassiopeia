#include "KElectrostaticAnalyticConicSectionIntegrator.hh"

#include "KElectrostaticAnalyticRingIntegrator.hh"
#include "KEllipticIntegrals.hh"
#include "KGaussianQuadrature.hh"

namespace KEMField
{
/**
 * \image html potentialFromConicSect.gif
 * Computes the electric potential at a point P (P[0],P[1],P[2]) due to the conic section by computing the following integrals:
 * \f[
 * V = \frac{\sigma}{\pi \epsilon_0} \cdot \int_{0}^{L} \frac{R(x) \cdot K(k(x))}{S(x)}\cdot dx,\\
 * \f]
 * where
 * \f{eqnarray*}{
 * R(x) = R_{a}+x \cdot \frac{R_b-R_a}{L},\\
 * Z(x) = Z_{a}+x \cdot \frac{Z_b-Z_a}{L},
 * \f}
 * and
 * \f{eqnarray*}{
 * r &=& \sqrt{(P[0])^2+(P[1])^2},\\
 * z &=& P[2],\\
 * S(Z) &=& \sqrt{(R+r)^2+(z-Z)^2},\\
 * k(Z) &=& \frac{2\sqrt{R \cdot r}}{S}.
 * \f}
 */
double KElectrostaticAnalyticConicSectionIntegrator::Potential(const KConicSection* source,const KPosition& P) const
{
	static const double ln4=1.386294361119890;

	static double (*f[1])(const double*,double*)
    		  = {&KElectrostaticAnalyticRingIntegrator::PotentialFromChargedRing};

	// integration parameters
	double par[7]; // par[0]: z  par[4]: zB
	// par[1]: r  par[5]: rB
	// par[2]: zA par[6]: L
	// par[3]: rA

	double za,ra,zb,rb,L,Da,Db,u[2],z,r,D;
	double q,pp,a,b,pmin,pmax,h;
	int n;

	z  = par[0]=P[2];
	r  = par[1]=sqrt(P[0]*P[0]+P[1]*P[1]);
	za = par[2]=source->GetZ0();
	ra = par[3]=source->GetR0();
	zb = par[4]=source->GetZ1();
	rb = par[5]=source->GetR1();
	L  = par[6]=(source->GetP0() - source->GetP1()).Magnitude();

	Da = sqrt((za-z)*(za-z)+(ra-r)*(ra-r));
	Db = sqrt((zb-z)*(zb-z)+(rb-r)*(rb-r));
	D  = fabs(Da+Db-L)/L;

	static KGaussianQuadrature Quad;

	if(D>=5.e-2)
		Quad(f,1,0.,L,par,20,&q);
	else if(D>=5.e-3 && D<5.e-2)
		Quad(f,1,0.,L,par,100,&q);
	else if(D>=5.e-4 && D<5.e-3)
		Quad(f,1,0.,L,par,500,&q);
	else
	{
		u[0]=(zb-za)/L;
		u[1]=(rb-ra)/L;
		pp = (z-za)*u[0] + (r-ra)*u[1];
		q=0.;
		if(pp < L)
		{
			pmax = L;
			pmin = pp;
			if(pp<0.)
				pmin=0.;
			b = pmax;
			a = pmin + (b-pmin)*0.3;
			double tmp;
			for(n=1;n<=35;n++)
			{
				Quad(f,1,a,b,par,50,&tmp);
				q+=tmp;
				if(fabs(a-pmin)/L<1.e-8) break;
				b=a;
				a=pmin+(b-pmin)*0.3;
			}
			h = fabs(a-pmin);
			q+=h*(ln4+log(2.*par[1]+1.e-12)+1.-log(h))/2.;
		}
		if(pp>0.)
		{
			pmax = pp;
			pmin = 0.;
			if(pp>L)
				pmax=L;
			b = pmin;
			a = pmax-(pmax-b)*0.3;
			double tmp;
			for(n=1;n<=35;n++)
			{
				Quad(f,1,b,a,par,50,&tmp);
				q+=tmp;
				if(fabs(pmax-a)/L<1.e-8) break;
				b=a;
				a=pmax-(pmax-b)*0.3;
			}
			h = fabs(pmax-a);
			q+=h*(ln4+log(2.*par[1]+1.e-12)+1.-log(h))/2.;
		}
	}

	return 1./(KEMConstants::Pi*KEMConstants::Eps0)*q;
}


KThreeVector KElectrostaticAnalyticConicSectionIntegrator::ElectricField(const KConicSection* source,const KPosition& P) const
{
	static const double ln4=1.386294361119890;

	static double (*f[2])(const double*,double*)
    		  = {&KElectrostaticAnalyticRingIntegrator::EFieldRFromChargedRing,
    				  &KElectrostaticAnalyticRingIntegrator::EFieldZFromChargedRing};

	// integration parameters
	double par[7]; // par[0]: z  par[4]: zB
	// par[1]: r  par[5]: rB
	// par[2]: zA par[6]: L
	// par[3]: rA

	double za,ra,zb,rb,L,Da,Db,u[2],z,r,D;
	double q[2],pp,a,b,pmin,pmax,h;
	int n;

	z  = par[0]=P[2];
	r  = par[1]=sqrt(P[0]*P[0]+P[1]*P[1]);
	za = par[2]=source->GetZ0();
	ra = par[3]=source->GetR0();
	zb = par[4]=source->GetZ1();
	rb = par[5]=source->GetR1();
	L  = par[6]=(source->GetP0() - source->GetP1()).Magnitude();

	Da = sqrt((za-z)*(za-z)+(ra-r)*(ra-r));
	Db = sqrt((zb-z)*(zb-z)+(rb-r)*(rb-r));
	D  = fabs(Da+Db-L)/L;

	static KGaussianQuadrature Quad;

	if(D>=5.e-2)
		Quad(f,2,0.,L,par,20,q);
	else if(D>=5.e-3 && D<5.e-2)
		Quad(f,2,0.,L,par,100,q);
	else if(D>=5.e-4 && D<5.e-3)
		Quad(f,2,0.,L,par,500,q);
	else
	{
		u[0]=(zb-za)/L;
		u[1]=(rb-ra)/L;
		pp = (z-za)*u[0] + (r-ra)*u[1];
		q[0] = q[1] = 0.;
		if(pp < L)
		{
			pmax = L;
			pmin = pp;
			if(pp<0.)
				pmin=0.;
			b = pmax;
			a = pmin + (b-pmin)*0.3;
			double tmp[2];
			for(n=1;n<=35;n++)
			{
				Quad(f,2,a,b,par,50,tmp);
				q[0]+=tmp[0];q[1]+=tmp[1];
				if(fabs(a-pmin)/L<1.e-8) break;
				b=a;
				a=pmin+(b-pmin)*0.3;
			}
			h = fabs(a-pmin);
			q[0]+=h*(ln4+log(2.*par[1]+1.e-12)+1.-log(h))/2.;
			q[1]+=h*(ln4+log(2.*par[1]+1.e-12)+1.-log(h))/2.;
		}
		if(pp>0.)
		{
			pmax = pp;
			pmin = 0.;
			if(pp>L)
				pmax=L;
			b = pmin;
			a = pmax-(pmax-b)*0.3;
			double tmp[2];
			for(n=1;n<=35;n++)
			{
				Quad(f,2,b,a,par,50,tmp);
				q[0]+=tmp[0];q[1]+=tmp[1];
				if(fabs(pmax-a)/L<1.e-8) break;
				b=a;
				a=pmax-(pmax-b)*0.3;
			}
			h = fabs(pmax-a);
			q[0]+=h*(ln4+log(2.*par[1]+1.e-12)+1.-log(h))/2.;
			q[1]+=h*(ln4+log(2.*par[1]+1.e-12)+1.-log(h))/2.;
		}
	}

	double Er = 1./(KEMConstants::Pi*KEMConstants::Eps0)*q[0];
	double Ez = 1./(KEMConstants::Pi*KEMConstants::Eps0)*q[1];

	KThreeVector field;
	field[2] = Ez;

	if (par[1]<1.e-14)
		field[0]=field[1]=0;
	else
	{
		double cosine = P[0]/par[1];
		double sine = P[1]/par[1];

		field[0] = cosine*Er;
		field[1] = sine*Er;
	}

	return field;
}

}
