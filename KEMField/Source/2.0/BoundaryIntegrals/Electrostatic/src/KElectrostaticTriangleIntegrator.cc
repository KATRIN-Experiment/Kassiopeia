#include "KElectrostaticTriangleIntegrator.hh"

#include "KDataDisplay.hh"

namespace KEMField
{

  /**
   * \image html potentialFromTriangle.gif
   * Returns the electric potential at a point P (P[0],P[1],P[2]) due to the 
   * collection of triangles.
   */
  double KElectrostaticTriangleIntegrator::Potential(const KTriangle* source,
						    const KPosition& P) const
  {
    double x_loc[3];
    double y_loc[2];
    double z_loc[1];
    double a_loc[2];
    double b_loc[2];
    double u_loc[2];

    KPosition p = source->GetP0() - P;

    double n1dotn2 = source->GetN1().Dot(source->GetN2());
    KDirection N2prime = (source->GetN2() - (n1dotn2*source->GetN1())).Unit();
    double n2dotn2prime = source->GetN2().Dot(N2prime);
    x_loc[0] = p.Dot(source->GetN1());
    y_loc[0] = p.Dot(N2prime);
    z_loc[0] = -p.Dot(source->GetN3());

    x_loc[1] = x_loc[0] + source->GetA();

    x_loc[2] = x_loc[0] + source->GetB()*n1dotn2;

    if (z_loc[0]<0)
    {
      y_loc[0] = -y_loc[0];
      y_loc[1] = y_loc[0] - source->GetB()*n2dotn2prime;
    }
    else
    {
      y_loc[1] = y_loc[0] + source->GetB()*n2dotn2prime;
    }

    z_loc[0] = fabs(z_loc[0]);

    if (z_loc[0]>1.e-14)
    {
      u_loc[0] = y_loc[0]/z_loc[0];
      if (u_loc[0]==0.) u_loc[0] = 1.e-18;
      u_loc[1] = y_loc[1]/z_loc[0];
      if (u_loc[1]==0.) u_loc[1] = 1.e-18;
      a_loc[0] = x_loc[0]/z_loc[0];
      a_loc[0] = (x_loc[0]*y_loc[1] - x_loc[2]*y_loc[0])/(z_loc[0]*(y_loc[1]-y_loc[0]));
      a_loc[1] = (x_loc[1]*y_loc[1] - x_loc[2]*y_loc[0])/(z_loc[0]*(y_loc[1]-y_loc[0]));
    }
    else
    {
      u_loc[0] = y_loc[0];
      if (u_loc[0]==0.) u_loc[0] = 1.e-18;
      u_loc[1] = y_loc[1];
      if (u_loc[1]==0.) u_loc[1] = 1.e-18;
      a_loc[0] = (x_loc[0]*y_loc[1] - x_loc[2]*y_loc[0])/(y_loc[1]-y_loc[0]);
      a_loc[1] = (x_loc[1]*y_loc[1] - x_loc[2]*y_loc[0])/(y_loc[1]-y_loc[0]);
    }
    b_loc[0] = (x_loc[2] - x_loc[0])/(y_loc[1] - y_loc[0]);
    b_loc[1] = (x_loc[2] - x_loc[1])/(y_loc[1] - y_loc[0]);

    double I = 0;

    if (z_loc[0]>1.e-14)
    {
      if (fabs(b_loc[0])<1.e-13)
	I = z_loc[0]*(I1(a_loc[1],b_loc[1],u_loc[0],u_loc[1]) - 
		      I2(a_loc[0],u_loc[0],u_loc[1]));
      else if (fabs(b_loc[1])<1.e-13)
	I = z_loc[0]*(I2(a_loc[1],u_loc[0],u_loc[1]) - 
		      I1(a_loc[0],b_loc[0],u_loc[0],u_loc[1]));
      else
	I = z_loc[0]*(I1(a_loc[1],b_loc[1],u_loc[0],u_loc[1]) - 
		      I1(a_loc[0],b_loc[0],u_loc[0],u_loc[1]));
    }
    else
    {
      I = (Potential_noZ(a_loc[1],b_loc[1],a_loc[0],b_loc[0],u_loc[1]) -
	   Potential_noZ(a_loc[1],b_loc[1],a_loc[0],b_loc[0],u_loc[0]));
    }

    I = fabs(I);

    return I/(4.*M_PI*KEMConstants::Eps0);
  }

  KEMThreeVector KElectrostaticTriangleIntegrator
  ::ElectricField(const KTriangle* source,
		  const KPosition& P) const
  {
    double x_loc[3];
    double y_loc[2];
    double z_loc[1];
    double a_loc[2];
    double b_loc[2];
    double u_loc[2];
    double z_sign;
    KEMThreeVector local_field;
    KEMThreeVector field;

    double dist = (source->Centroid()-P).Magnitude();

    KPosition p = source->GetP0() - P;

    double n1dotn2 = source->GetN1().Dot(source->GetN2());
    KDirection N2prime = (source->GetN2() - (n1dotn2*source->GetN1())).Unit();
    double n2dotn2prime = source->GetN2().Dot(N2prime);
    x_loc[0] = p.Dot(source->GetN1());
    y_loc[0] = p.Dot(N2prime);
    z_loc[0] = -(p.Dot(source->GetN3()));

    x_loc[1] = x_loc[0] + source->GetA();

    x_loc[2] = x_loc[0] + source->GetB()*n1dotn2;

    if (z_loc[0]<0)
    {
      y_loc[0] = -y_loc[0];
      y_loc[1] = y_loc[0] - source->GetB()*n2dotn2prime;
      z_sign = -1;
    }
    else
    {
      y_loc[1] = y_loc[0] + source->GetB()*n2dotn2prime;
      z_sign = 1;
    }

    z_loc[0] = fabs(z_loc[0]);

    if (z_loc[0]>1.e-14)
    {
      u_loc[0] = y_loc[0]/z_loc[0];
      if (u_loc[0]==0.) u_loc[0] = 1.e-18;
      u_loc[1] = y_loc[1]/z_loc[0];
      if (u_loc[1]==0.) u_loc[1] = 1.e-18;
      a_loc[0] = x_loc[0]/z_loc[0];
      a_loc[0] = (x_loc[0]*y_loc[1] - x_loc[2]*y_loc[0])/(z_loc[0]*(y_loc[1]-y_loc[0]));
      a_loc[1] = (x_loc[1]*y_loc[1] - x_loc[2]*y_loc[0])/(z_loc[0]*(y_loc[1]-y_loc[0]));
    }
    else
    {
      u_loc[0] = y_loc[0];
      if (u_loc[0]==0.) u_loc[0] = 1.e-18;
      u_loc[1] = y_loc[1];
      if (u_loc[1]==0.) u_loc[1] = 1.e-18;
      a_loc[0] = (x_loc[0]*y_loc[1] - x_loc[2]*y_loc[0])/(y_loc[1]-y_loc[0]);
      a_loc[1] = (x_loc[1]*y_loc[1] - x_loc[2]*y_loc[0])/(y_loc[1]-y_loc[0]);
    }
    b_loc[0] = (x_loc[2] - x_loc[0])/(y_loc[1] - y_loc[0]);
    b_loc[1] = (x_loc[2] - x_loc[1])/(y_loc[1] - y_loc[0]);

    double prefac = 1./(4.*KEMConstants::Pi*KEMConstants::Eps0);

    if (z_loc[0]>1.e-14)
    {
      local_field[0] = z_sign*prefac*Local_Ex(a_loc[0],a_loc[1],
					      b_loc[0],b_loc[1],
					      u_loc[0],u_loc[1]);
      local_field[1] = prefac*Local_Ey(a_loc[0],a_loc[1],
				       b_loc[0],b_loc[1],
				       u_loc[0],u_loc[1]);
      local_field[2] = prefac*Local_Ez(a_loc[0],a_loc[1],
				       b_loc[0],b_loc[1],
				       u_loc[0],u_loc[1]);
    }
    else
    {
      local_field[0] = z_sign*prefac*Local_Ex(a_loc[0],a_loc[1],
					      b_loc[0],b_loc[1],
					      u_loc[0],u_loc[1]);
      local_field[1] = prefac*Local_Ey(a_loc[0],a_loc[1],
				       b_loc[0],b_loc[1],
				       u_loc[0],u_loc[1]);
      if (dist<1.e-12)
	local_field[2] = 1./(2.*KEMConstants::Eps0);
      else
	local_field[2] = 0.;
    }

    for (int j=0;j<3;j++)
    {
      field[j] = (source->GetN1()[j]*local_field[0] +
		  N2prime[j]*local_field[1] + 
		  source->GetN3()[j]*local_field[2]);
    }

    return field;
  }

  double KElectrostaticTriangleIntegrator::Potential_noZ(double a2,
							   double b2,
							   double a1,
							   double b1,
							   double y) const
  {
    double logArg2 = (1.+b2*b2)*y+a2*b2+sqrt(1.+b2*b2)*sqrt((1.+b2*b2)*y*y+2*a2*b2*y+a2*a2);

    double logArg1 = (1.+b1*b1)*y+a1*b1+sqrt(1.+b1*b1)*sqrt((1.+b1*b1)*y*y+2*a1*b1*y+a1*a1);

    double ans2 = 0;

    if (logArg2>1.e-14)
    {
      if (fabs(y)>1.e-14)
	ans2 = y*asinh((a2+b2*y)/fabs(y)) + a2/sqrt(1.+b2*b2)*log(logArg2);
      else
	ans2 = a2/sqrt(1.+b2*b2)*log(logArg2);
    }
    else
    {
      if (fabs(y)>1.e-14)
	ans2 = y*asinh(y*b2/fabs(y));
      else
	ans2 = 0.;
    }

    double ans1 = 0.;

    if (logArg1>1.e-14)
    {
      if (fabs(y)>5.e-14)
	ans1 = y*asinh((a1+b1*y)/fabs(y)) + a1/sqrt(1.+b1*b1)*log(logArg1);
      else
	ans1 = a1/sqrt(1.+b1*b1)*log(logArg1);      
    }
    else
    {
      if (fabs(y)>5.e-14)
	ans1 = y*asinh(y*b1/fabs(y));
      else
	ans1 = 0.;
    }
    return ans2-ans1;
  }

  double KElectrostaticTriangleIntegrator::F1(double a,double b,double u) const
  {
    return u*asinh((a + b*u)/sqrt(u*u+1.));
  }

  double KElectrostaticTriangleIntegrator::I3(double a,double b,double u1,double u2) const
  {
    double g1 = (sqrt(b*b+1.)*sqrt(a*a+2*a*b*u1+(b*b+1.)*u1*u1+1.)+b*(a+b*u1)+u1);
    double g2 = (sqrt(b*b+1.)*sqrt(a*a+2*a*b*u2+(b*b+1.)*u2*u2+1.)+b*(a+b*u2)+u2);

    if (fabs(g1)<1.e-12)
    {
      if (fabs(a)<1.e-14)
	g1 = 1.e-12;
    }
    if (fabs(g2)<1.e-12)
    {
      if (fabs(a)<1.e-14)
	g2 = 1.e-12;
    }

    return a/sqrt(b*b+1.)*log(g2/g1);
  }

  double KElectrostaticTriangleIntegrator::I3p(double a,double b,double u1,double u2) const
  {
    double g1 = (sqrt(b*b+1.)*sqrt(a*a+2*a*b*u1+(b*b+1.)*u1*u1+1.)+b*(a+b*u1)+u1);
    double g2 = (sqrt(b*b+1.)*sqrt(a*a+2*a*b*u2+(b*b+1.)*u2*u2+1.)+b*(a+b*u2)+u2);

    return 1./sqrt(b*b+1.)*log(g2/g1);
  }

  double KElectrostaticTriangleIntegrator::I4(double alpha,
						double gamma,
						double q2,
						double prefac,
						double t1,
						double t2) const
  {
    // double q = sqrt(gamma-alpha);
    double q  = sqrt(q2);
    double g1 = sqrt(gamma*t1*t1 + alpha);
    double g2 = sqrt(gamma*t2*t2 + alpha);

    if (t1>1.e15 || t2>1.e15)
    {
      if (t2<1.e15)
	return (prefac*1./q*(atan(g2/q)-KEMConstants::PiOverTwo));
      else if (t1<1.e15)
	return (prefac*1./q*(KEMConstants::PiOverTwo-atan(g1/q)));
      else
	return 0.;
    }

    return prefac*1./q*atan(q*(g2-g1)/(q2+g1*g2));
  }

  double KElectrostaticTriangleIntegrator::I4(double a,double b,double u1,double u2) const
  {
    if (fabs(u1-b/a)<1.e-14)
    {
      if (u2 > b/a)
	return I4(a,b,u1+1.e-14,u2);
      else
	return I4(a,b,u1-1.e-14,u2);
    }

    if (fabs(u2-b/a)<1.e-14)
    {
      if (u1 > b/a)
	return I4(a,b,u1,u2+1.e-14);
      else
	return I4(a,b,u1,u2-1.e-14);
    }

    if (fabs(a)<1.e-14)
    {
      if (a>0)
	return I4(a+1.e-14,b,u1,u2);
      else
	return I4(a-1.e-14,b,u1,u2);
    }

    double alpha = ((double)1.) + (a*a)/(b*b);
    double gamma = (a*a+b*b)*(a*a+b*b+((double)1.))/(b*b);
    // q^2 = (gamma - alpha) has been added to dodge roundoff error (08/28/12)
    double q2 = (a*a+b*b)*(a*a+b*b)/(b*b);
    double prefac = (a*a/b + b);
    double t1;
    if (a*u1!=b)
      t1 = (b*u1 + a)/(a*u1 - b);
    else
      t1 = 1.e15;
    double t2;
    if (a*u2!=b)
      t2 = (b*u2 + a)/(a*u2 - b);
    else
      t2 = 1.e15;

    double sign = 1.;

    if (a<0.)
      sign=-sign;
    if (b<0.)
      sign=-sign;
    if (u1>b/a)
      sign=-sign;

    // if the function diverges within our region of integration, we must cut out
    // the divergence
    if (((u1 > b/a) - (u1 < b/a)) != ((u2 > b/a) - (u2 < b/a)))
      return sign*(I4(alpha,gamma,q2,prefac,t1,1.e16) +
		   I4(alpha,gamma,q2,prefac,t2,1.e16));
    else
      return sign*I4(alpha,gamma,q2,prefac,t1,t2);
  }

  double KElectrostaticTriangleIntegrator::I4_2(double alpha,
						  double gamma,
						  double prefac,
						  double t1,
						  double t2) const
  {
    double g1 = sqrt(alpha*t1*t1 + gamma);
    double g2 = sqrt(alpha*t2*t2 + gamma);
    double q = sqrt(gamma-alpha);

    return prefac*1./q*atanh(q*(g2-g1)/((alpha-gamma)+g1*g2));
  }

  double KElectrostaticTriangleIntegrator::I4_2(double a,double b,double u1,double u2) const
  {
    double alpha = ((double)1.) + (a*a)/(b*b);
    double gamma = (a*a+b*b)*(a*a+b*b+1)/(b*b);
    double lambda = -a/b;
    double prefac = (a*a/b + b);
    double t1;
    if (b*u1!=-a)
      t1 = (b - a*u1)/(a + b*u1);
    else
      t1 = 1.e15;
    double t2;
    if (b*u2!=-a)
      t2 = (b - a*u2)/(a + b*u2);
    else
      t2 = 1.e15;

    // if the function diverges within our region of integration, we must cut out
    // the divergence
    if (((u1 > lambda) - (u1 < lambda)) != ((u2 > lambda) - (u2 < lambda)))
    {
      if (u1>lambda)
	return (I4_2(alpha,gamma,prefac,fabs(t1),1.e15) +
		I4_2(alpha,gamma,prefac,fabs(t2),1.e15));
      else
	return (I4_2(alpha,gamma,prefac,1.e15,fabs(t1)) +
		I4_2(alpha,gamma,prefac,1.e15,fabs(t2)));
    }
    else if (u1>lambda)
      return I4_2(alpha,gamma,prefac,t1,t2);
    else
      return I4_2(alpha,gamma,prefac,t2,t1);
  }

  double KElectrostaticTriangleIntegrator::I1(double a,double b,double u1,double u2) const
  {
    return F1(a,b,u2) - F1(a,b,u1) + I3(a,b,u1,u2) - I4(a,b,u1,u2);
  }

  double KElectrostaticTriangleIntegrator::I6(double x,double u1, double u2) const
  {
    if (fabs(x)<1.e-15)
      return 0;
    return x*log((sqrt(u2*u2+x*x+1.)+u2)/(sqrt(u1*u1+x*x+1.)+u1));
  }

  double KElectrostaticTriangleIntegrator::I7(double x,double u1,double u2) const
  {
    double t1;
    if (fabs(u1)>1.e-16)
      t1 = 1./u1;
    else
      t1 = 1.e16;
    double t2;
    if (fabs(u2)>1.e-16)
      t2 = 1./u2;
    else
      t2 = 1.e16;

    double g1 = sqrt(1.+t1*t1*(1.+x*x));
    double g2 = sqrt(1.+t2*t2*(1.+x*x));

    return atan(x*(g2-g1)/(x*x+g2*g1));
  }

  double KElectrostaticTriangleIntegrator::I2(double x,double u1,double u2) const
  {
    double ans = 0.;

    if (((u1 > 0.) - (u1 < 0.)) != ((u2 > 0.) - (u2 < 0.)))
    {
      if (u1<=0.)
	ans = (F1(x,0.,u2) - F1(x,0.,u1)) + I6(x,u1,u2) + I7(x,0.,fabs(u1)) + I7(x,0.,fabs(u2));
      else
	ans = (F1(x,0.,u2) - F1(x,0.,u1)) + I6(x,u1,u2) + I7(x,fabs(u1),0.) + I7(x,fabs(u2),0.);
    }
    else if (u1<=0.)
      ans = (F1(x,0.,u2) - F1(x,0.,u1)) + I6(x,u1,u2) + I7(x,u2,u1);
    else
      ans = (F1(x,0.,u2) - F1(x,0.,u1)) + I6(x,u1,u2) + I7(x,u1,u2);

    return ans;
  }

  double KElectrostaticTriangleIntegrator::J2(double a,double u1,double u2) const
  {
    if (a==0.)
      return 0.;

    double g1 = sqrt(u1*u1+a*a+1.);
    double g2 = sqrt(u2*u2+a*a+1.);

    return a/(2.*fabs(a))*log(((g2-fabs(a))*(g1+fabs(a)))/
			      ((g2+fabs(a))*(g1-fabs(a))));
  }

  double KElectrostaticTriangleIntegrator::Local_Ex(double a0,double a1,
						      double b0,double b1,
						      double u0,double u1) const
  {
    double ans = (I3p(a1,b1,u0,u1) - I3p(a0,b0,u0,u1));

    return ans;
  }

  double KElectrostaticTriangleIntegrator::Local_Ey(double a0,double a1,
						      double b0,double b1,
						      double u0,double u1) const
  {
    double I2 = 0.;
    if (fabs(b1)>1.e-14)
      I2 = I4_2(a1,b1,u0,u1) + b1*I3p(a1,b1,u0,u1);
    else
      I2 = J2(a1,u0,u1);

    double I1 = 0.;
    if (fabs(b0)>1.e-14)
      I1 = I4_2(a0,b0,u0,u1) + b0*I3p(a0,b0,u0,u1);
    else
      I1 = J2(a0,u0,u1);

    return I1-I2;
  }

  double KElectrostaticTriangleIntegrator::Local_Ez(double a0,double a1,
						      double b0,double b1,
						      double u0,double u1) const
  {
    double I1 = 0.;
    if (fabs(b0)>1.e-14)
      I1 = I4(a0,b0,u0,u1);
    else
      if (((u0 > 0.) - (u0 < 0.)) != ((u1 > 0.) - (u1 < 0.)))
      {
	if (u0<=0.)
	  I1 = -(I7(a0,0.,fabs(u0)) + I7(a0,0.,fabs(u1)));
	else
	  I1 = -(I7(a0,fabs(u0),0.) + I7(a0,fabs(u1),0.));
      }
      else if (u0<=0.)
	I1 = I7(a0,u0,u1);
      else
	I1 = I7(a0,u1,u0);

    double I2 = 0.;
    if (fabs(b1)>1.e-14)
      I2 = I4(a1,b1,u0,u1);
    else
      if (((u0 > 0.) - (u0 < 0.)) != ((u1 > 0.) - (u1 < 0.)))
      {
	if (u0<=0.)
	  I2 = -(I7(a1,0.,fabs(u0)) + I7(a1,0.,fabs(u1)));
	else
	  I2 = -(I7(a1,fabs(u0),0.) + I7(a1,fabs(u1),0.));
      }
      else if (u0<=0.)
	I2 = I7(a1,u0,u1);
      else
	I2 = I7(a1,u1,u0);

    double ans2 = I2-I1;

    return ans2;
  }

    double KElectrostaticTriangleIntegrator::Potential(const KSymmetryGroup<KTriangle>* source, const KPosition& P) const
    {
      double potential = 0.;
      for (KSymmetryGroup<KTriangle>::ShapeCIt it=source->begin();it!=source->end();++it)
	potential += Potential(*it,P);
      return potential;
    }

    KEMThreeVector KElectrostaticTriangleIntegrator::ElectricField(const KSymmetryGroup<KTriangle>* source, const KPosition& P) const
    {
      KEMThreeVector electricField(0.,0.,0.);
      for (KSymmetryGroup<KTriangle>::ShapeCIt it=source->begin();it!=source->end();++it)
	electricField += ElectricField(*it,P);
      return electricField;
    }
}
