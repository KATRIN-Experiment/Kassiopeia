#ifndef KEMFIELD_ELECTROSTATICTRIANGLE_CUH
#define KEMFIELD_ELECTROSTATICTRIANGLE_CUH

#include <cmath>

// Triangle geometry definition (as defined by the streamers in KTriangle.hh):
//
// data[0]:     A
// data[1]:     B
// data[2..4]:  P0[0..2]
// data[5..7]:  N1[0..2]
// data[8..10]: N2[0..2]

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE ArcSinh(CU_TYPE x)
{
  // On some systems, ArcSinh(-x) will return -inf much sooner than ArcSinh(x)
  // returns +inf.  So, we always compute the positive version.
  CU_TYPE prefac = 1.;
  if( signbit(x) )
  {
    prefac = -1.;
    x = -x;
  }
  return prefac*LOG(x + SQRT(1. + x*x));
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE ArcTanh(CU_TYPE x)
{
  return .5*LOG((1. + x)/(1. - x));
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE ET_Potential_noZ(CU_TYPE a2,
                        CU_TYPE b2,
                        CU_TYPE a1,
                        CU_TYPE b1,
                        CU_TYPE y)
{
  CU_TYPE logArg2 = (1+b2*b2)*y+a2*b2+SQRT(1+b2*b2)*SQRT((1+b2*b2)*y*y+2*a2*b2*y+a2*a2);

  CU_TYPE logArg1 = (1+b1*b1)*y+a1*b1+SQRT(1+b1*b1)*SQRT((1+b1*b1)*y*y+2*a1*b1*y+a1*a1);

  CU_TYPE ans2 = 0;

  if (logArg2>1.e-14)
  {
    if (FABS(y)>1.e-14)
      ans2 = y*ArcSinh((a2+b2*y)/FABS(y)) + a2/SQRT(1+b2*b2)*LOG(logArg2);
    else
      ans2 = a2/SQRT(1+b2*b2)*LOG(logArg2);
  }
  else
  {
    if (FABS(y)>1.e-14)
      ans2 = y*ArcSinh(y*b2/FABS(y));
    else
      ans2 = 0.;
  }

  CU_TYPE ans1 = 0;

  if (logArg1>1.e-14)
  {
    if (FABS(y)>5.e-14)
      ans1 = y*ArcSinh((a1+b1*y)/FABS(y)) + a1/SQRT(1+b1*b1)*LOG(logArg1);
    else
      ans1 = a1/SQRT(1+b1*b1)*LOG(logArg1);
  }
  else
  {
    if (FABS(y)>5.e-14)
      ans1 = y*ArcSinh(y*b1/FABS(y));
    else
      ans1 = 0.;
  }
  return ans2-ans1;
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE ET_F1(CU_TYPE a,CU_TYPE b,CU_TYPE u)
{
  return u*ArcSinh((a + b*u)/SQRT(u*u+1));
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE ET_I3_(CU_TYPE a,CU_TYPE b,CU_TYPE u1,CU_TYPE u2)
{
  CU_TYPE g1;
  CU_TYPE g2;

  g1 = (SQRT(b*b+1.)*SQRT(a*a+2*a*b*u1+(b*b+1.)*u1*u1+1.)+b*(a+b*u1)+u1);
  g2 = (SQRT(b*b+1.)*SQRT(a*a+2*a*b*u2+(b*b+1.)*u2*u2+1.)+b*(a+b*u2)+u2);

  //the following two lines are a patch to fix an error due floating point
  //rounding which results in a negative value of g2/g1, by R. Combe 2/2/15
  if(g1 <= 0){g1 = -(1+a*a+b*b)/(2*(b*b+1)*u1);};
  if(g2 <= 0){g2 = -(1+a*a+b*b)/(2*(b*b+1)*u2);};

  if (FABS(g1)<1.e-12)
  {
    if (FABS(a)<1.e-14)
      g1 = 1.e-12;
  }
  if (FABS(g2)<1.e-12)
  {
    if (FABS(a)<1.e-14)
      g2 = 1.e-12;
  }

  //adding fabs to the log argument to catch small negative arguments
  //that still might slip through due to floating point math errors 2/2/15
  return a/SQRT(b*b+1.)*LOG(FABS(g2/g1));
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE ET_I3p_(CU_TYPE a,CU_TYPE b,CU_TYPE u1,CU_TYPE u2)
{
  CU_TYPE g1 = (SQRT(b*b+1)*SQRT(a*a+2*a*b*u1+(b*b+1)*u1*u1+1)+b*(a+b*u1)+u1);
  CU_TYPE g2 = (SQRT(b*b+1)*SQRT(a*a+2*a*b*u2+(b*b+1)*u2*u2+1)+b*(a+b*u2)+u2);

  //the following two lines are a patch to fix an error due floating point
  //rounding which results in a negative value of g2/g1, by R. Combe 2/2/15
  if(g1<=0){g1 = -(1+a*a+b*b)/(2*(b*b+1)*u1);};
  if(g2<=0){g2 = -(1+a*a+b*b)/(2*(b*b+1)*u2);};

  //adding fabs to the log argument to catch small negative arguments
  //that still might slip through due to floating point math errors 2/2/15
  return 1./SQRT(b*b+1.)*LOG(FABS(g2/g1));
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE ET_I4_2(CU_TYPE alpha,
		 CU_TYPE gamma,
		 CU_TYPE q2,
		 CU_TYPE prefac,
		 CU_TYPE t1,
		 CU_TYPE t2)
{
  // CU_TYPE q = SQRT(gamma-alpha);
  CU_TYPE q  = SQRT(q2);
  CU_TYPE g1 = SQRT(gamma*t1*t1 + alpha);
  CU_TYPE g2 = SQRT(gamma*t2*t2 + alpha);

  if (t1>1.e15 || t2>1.e15)
  {
    if (t2<1.e15)
      return (prefac*1./q*(ATAN(g2/q)-M_PI_OVER_2));
    else if (t1<1.e15)
      return (prefac*1./q*(M_PI_OVER_2-ATAN(g1/q)));
    else
      return 0.;
  }

  // return prefac*1./q*atan(q*(g2-g1)/((gamma-alpha)+g1*g2));
  return prefac*1./q*ATAN(q*(g2-g1)/(q2+g1*g2));
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE ET_I4_(CU_TYPE a,CU_TYPE b,CU_TYPE u1,CU_TYPE u2)
{
  CU_TYPE alpha = ((CU_TYPE)1.) + (a*a)/(b*b);
  CU_TYPE gamma = (a*a+b*b)*(a*a+b*b+((CU_TYPE)1.))/(b*b);
  // q^2 = (gamma - alpha) has been added to dodge roundoff error (08/28/12)
  CU_TYPE q2 = (a*a+b*b)*(a*a+b*b)/(b*b);
  CU_TYPE prefac = (a*a/b + b);
  CU_TYPE t1;
  if (a*u1!=b)
    t1 = (b*u1 + a)/(a*u1 - b);
  else
    t1 = 1.e15;
  CU_TYPE t2;
  if (a*u2!=b)
    t2 = (b*u2 + a)/(a*u2 - b);
  else
    t2 = 1.e15;

  CU_TYPE sign = 1.;

  if( signbit(a) )
    sign=-sign;
  if( signbit(b) )
    sign=-sign;
  if( u1>b/a )
    sign=-sign;

  // if the function diverges within our region of integration, we must cut out
  // the divergence
  if (((u1 > b/a) - (u1 < b/a)) != ((u2 > b/a) - (u2 < b/a)))
    return sign*(ET_I4_2(alpha,gamma,q2,prefac,t1,1.e16) +
		 ET_I4_2(alpha,gamma,q2,prefac,t2,1.e16));
  else
    return sign*ET_I4_2(alpha,gamma,q2,prefac,t1,t2);
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE ET_I4_2_2(CU_TYPE alpha,
		   CU_TYPE gamma,
		   CU_TYPE prefac,
		   CU_TYPE t1,
		   CU_TYPE t2)
{
  CU_TYPE g1 = SQRT(alpha*t1*t1 + gamma);
  CU_TYPE g2 = SQRT(alpha*t2*t2 + gamma);
  CU_TYPE q = SQRT(gamma-alpha);

  return prefac*1./q*ArcTanh(q*(g2-g1)/((alpha-gamma)+g1*g2));
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE ET_I4_2_(CU_TYPE a,CU_TYPE b,CU_TYPE u1,CU_TYPE u2)
{
  CU_TYPE alpha = 1 + (a*a)/(b*b);
  CU_TYPE gamma = (a*a+b*b)*(a*a+b*b+1)/(b*b);
  CU_TYPE lambda = -a/b;
  CU_TYPE prefac = (a*a/b + b);
  CU_TYPE t1;
  if (b*u1!=-a)
    t1 = (b - a*u1)/(a + b*u1);
  else
    t1 = 1.e15;
  CU_TYPE t2;
  if (b*u2!=-a)
    t2 = (b - a*u2)/(a + b*u2);
  else
    t2 = 1.e15;

  // if the function diverges within our region of integration, we must cut out
  // the divergence
  if (((u1 > lambda) - (u1 < lambda)) != ((u2 > lambda) - (u2 < lambda)))
  {
    if (u1>lambda)
      return (ET_I4_2_2(alpha,gamma,prefac,FABS(t1),1.e15) +
	      ET_I4_2_2(alpha,gamma,prefac,FABS(t2),1.e15));
    else
      return (ET_I4_2_2(alpha,gamma,prefac,1.e15,FABS(t1)) +
	      ET_I4_2_2(alpha,gamma,prefac,1.e15,FABS(t2)));
  }
  else if (u1>lambda)
    return ET_I4_2_2(alpha,gamma,prefac,t1,t2);
  else
    return ET_I4_2_2(alpha,gamma,prefac,t2,t1);
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE ET_I1_(CU_TYPE a,CU_TYPE b,CU_TYPE u1,CU_TYPE u2)
{
  return ET_F1(a,b,u2) - ET_F1(a,b,u1) + ET_I3_(a,b,u1,u2) - ET_I4_(a,b,u1,u2);
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE ET_I6_(CU_TYPE x,CU_TYPE u1, CU_TYPE u2)
{
  if (FABS(x)<1.e-15)
    return 0;
  return x*LOG((SQRT(u2*u2+x*x+1)+u2)/(SQRT(u1*u1+x*x+1)+u1));
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE ET_I7_(CU_TYPE x,CU_TYPE u1,CU_TYPE u2)
{
  CU_TYPE t1;
  if (FABS(u1)>1.e-16)
    t1 = 1./u1;
  else
    t1 = 1.e16;
  CU_TYPE t2;
  if (FABS(u2)>1.e-16)
    t2 = 1./u2;
  else
    t2 = 1.e16;

  CU_TYPE g1 = SQRT(1+t1*t1*(1+x*x));
  CU_TYPE g2 = SQRT(1+t2*t2*(1+x*x));

  return ATAN(x*(g2-g1)/(x*x+g2*g1));
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE ET_I2_(CU_TYPE x,CU_TYPE u1,CU_TYPE u2)
{
  CU_TYPE ans = 0.;

  if (((u1 > 0) - (u1 < 0)) != ((u2 > 0) - (u2 < 0)))
  {
    if (u1<=0)
      ans = (ET_F1(x,0,u2) - ET_F1(x,0,u1)) + ET_I6_(x,u1,u2) + ET_I7_(x,0,FABS(u1)) + ET_I7_(x,0,FABS(u2));
    else
      ans = (ET_F1(x,0,u2) - ET_F1(x,0,u1)) + ET_I6_(x,u1,u2) + ET_I7_(x,FABS(u1),0) + ET_I7_(x,FABS(u2),0);
  }
  else if (u1<=0)
    ans = (ET_F1(x,0,u2) - ET_F1(x,0,u1)) + ET_I6_(x,u1,u2) + ET_I7_(x,u2,u1);
  else
    ans = (ET_F1(x,0,u2) - ET_F1(x,0,u1)) + ET_I6_(x,u1,u2) + ET_I7_(x,u1,u2);

  return ans;
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE ET_J2_(CU_TYPE a,CU_TYPE u1,CU_TYPE u2)
{
  if (a==0)
    return 0.;

  CU_TYPE g1 = SQRT(u1*u1+a*a+1.);
  CU_TYPE g2 = SQRT(u2*u2+a*a+1.);

  return a/(2.*FABS(a))*LOG(((g2-FABS(a))*(g1+FABS(a)))/
  			    ((g2+FABS(a))*(g1-FABS(a))));
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE ET_Local_Ex(CU_TYPE a0,CU_TYPE a1,
		     CU_TYPE b0,CU_TYPE b1,
		     CU_TYPE u0,CU_TYPE u1)
{
  CU_TYPE ans = (ET_I3p_(a1,b1,u0,u1) - ET_I3p_(a0,b0,u0,u1));

  return ans;
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE ET_Local_Ey(CU_TYPE a0,CU_TYPE a1,
                    CU_TYPE b0,CU_TYPE b1,
                    CU_TYPE u0,CU_TYPE u1)
{
  CU_TYPE I2 = 0.;
  if (FABS(b1)>1.e-14)
    I2 = ET_I4_2_(a1,b1,u0,u1) + b1*ET_I3p_(a1,b1,u0,u1);
  else
    I2 = ET_J2_(a1,u0,u1);

  CU_TYPE I1 = 0.;
  if (FABS(b0)>1.e-14)
    I1 = ET_I4_2_(a0,b0,u0,u1) + b0*ET_I3p_(a0,b0,u0,u1);
  else
    I1 = ET_J2_(a0,u0,u1);

  return I1-I2;
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE ET_Local_Ez(CU_TYPE a0,CU_TYPE a1,
                    CU_TYPE b0,CU_TYPE b1,
                    CU_TYPE u0,CU_TYPE u1)
{
  CU_TYPE I1 = 0.;
  if (FABS(b0)>1.e-14)
    I1 = ET_I4_(a0,b0,u0,u1);
  else
    if (((u0 > 0) - (u0 < 0)) != ((u1 > 0) - (u1 < 0)))
    {
      if (u0<=0)
	I1 = -(ET_I7_(a0,0.,FABS(u0)) + ET_I7_(a0,0.,FABS(u1)));
      else
	I1 = -(ET_I7_(a0,FABS(u0),0.) + ET_I7_(a0,FABS(u1),0.));
    }
    else if (u0<=0)
      I1 = ET_I7_(a0,u0,u1);
    else
      I1 = ET_I7_(a0,u1,u0);

  CU_TYPE I2 = 0.;
  if (FABS(b1)>1.e-14)
    I2 = ET_I4_(a1,b1,u0,u1);
  else
    if (((u0 > 0) - (u0 < 0)) != ((u1 > 0) - (u1 < 0)))
    {
      if (u0<=0)
	I2 = -(ET_I7_(a1,0.,FABS(u0)) + ET_I7_(a1,0.,FABS(u1)));
      else
	I2 = -(ET_I7_(a1,FABS(u0),0.) + ET_I7_(a1,FABS(u1),0.));
    }
    else if (u0<=0)
      I2 = ET_I7_(a1,u0,u1);
    else
      I2 = ET_I7_(a1,u1,u0);

  CU_TYPE ans2 = I2-I1;

  return ans2;
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE ET_Potential(const CU_TYPE* P, const CU_TYPE* data)
{
  // Triangle calculation

  CU_TYPE x_loc[3];
  CU_TYPE y_loc[2];
  CU_TYPE z_loc[1];
  CU_TYPE a_loc[2];
  CU_TYPE b_loc[2];
  CU_TYPE u_loc[2];

  CU_TYPE n1dotn2 = 0;
  CU_TYPE n2dotn2prime = 0;

  x_loc[0] = 0;
  y_loc[0] = 0;
  z_loc[0] = 0;

  for (int j=0;j<3;j++)
    n1dotn2 += data[5 + j]*data[8 + j];

  CU_TYPE N2mag = 0.;
  CU_TYPE N2prime[3];
  for (int j=0;j<3;j++)
  {
    N2prime[j] = data[j+8] - n1dotn2*data[j+5];
    N2mag += N2prime[j]*N2prime[j];
  }
  N2mag = SQRT(N2mag);

  for (int j=0;j<3;j++)
    N2prime[j] = N2prime[j]/N2mag;

  CU_TYPE norm[3];
  Tri_Normal(norm,data);

  for (int j=0;j<3;j++)
  {
    n2dotn2prime += data[8 + j]*N2prime[j];
    x_loc[0] += (data[2 + j]-P[j])*data[5 + j];
    y_loc[0] += (data[2 + j]-P[j])*N2prime[j];
    z_loc[0] += (P[j]-data[2 + j])*norm[j];
  }

  x_loc[1] = x_loc[0] + data[0];

  x_loc[2] = x_loc[0] + data[1]*n1dotn2;

  if (z_loc[0]<0)
  {
    y_loc[0] = -y_loc[0];
    y_loc[1] = y_loc[0] - data[1]*n2dotn2prime;
  }
  else
  {
    y_loc[1] = y_loc[0] + data[1]*n2dotn2prime;
  }

  z_loc[0] = FABS(z_loc[0]);

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

  CU_TYPE I = 0;

  if (z_loc[0]>1.e-14)
  {
    if (FABS(b_loc[0])<1.e-13)
    {
      I = z_loc[0]*(ET_I1_(a_loc[1],b_loc[1],u_loc[0],u_loc[1]) -
		    ET_I2_(a_loc[0],u_loc[0],u_loc[1]));
    }
    else if (FABS(b_loc[1])<1.e-13)
    {
      I = z_loc[0]*(ET_I2_(a_loc[1],u_loc[0],u_loc[1]) -
		    ET_I1_(a_loc[0],b_loc[0],u_loc[0],u_loc[1]));
    }
    else
    {
      I = z_loc[0]*(ET_I1_(a_loc[1],b_loc[1],u_loc[0],u_loc[1]) -
		    ET_I1_(a_loc[0],b_loc[0],u_loc[0],u_loc[1]));
    }
  }
  else
  {
    I = (ET_Potential_noZ(a_loc[1],b_loc[1],a_loc[0],b_loc[0],u_loc[1]) -
	 ET_Potential_noZ(a_loc[1],b_loc[1],a_loc[0],b_loc[0],u_loc[0]));
  }

  I = FABS(I);

  return I/(4.*M_PI*M_EPS0);
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE4 ET_EField(const CU_TYPE* P, const CU_TYPE* data)
{
  // Triangle calculation

  CU_TYPE x_loc[3];
  CU_TYPE y_loc[2];
  CU_TYPE z_loc[1];
  CU_TYPE a_loc[2];
  CU_TYPE b_loc[2];
  CU_TYPE u_loc[2];
  CU_TYPE z_sign;
  CU_TYPE local_field[3];
  CU_TYPE field_[3];

  CU_TYPE n1dotn2 = 0;
  CU_TYPE n2dotn2prime = 0;

  x_loc[0] = 0;
  y_loc[0] = 0;
  z_loc[0] = 0;

  for (int j=0;j<3;j++)
    n1dotn2 += data[5 + j]*data[8 + j];

  CU_TYPE N2mag = 0.;
  CU_TYPE N2prime[3];
  for (int j=0;j<3;j++)
  {
    N2prime[j] = data[j+8] - n1dotn2*data[j+5];
    N2mag += N2prime[j]*N2prime[j];
  }
  N2mag = SQRT(N2mag);

  for (int j=0;j<3;j++)
    N2prime[j] = N2prime[j]/N2mag;

  CU_TYPE norm[3];
  Tri_Normal(norm,data);

  for (int j=0;j<3;j++)
  {
    n2dotn2prime += data[8 + j]*N2prime[j];
    x_loc[0] += (data[2 + j]-P[j])*data[5 + j];
    y_loc[0] += (data[2 + j]-P[j])*N2prime[j];
    z_loc[0] += (P[j]-data[2 + j])*norm[j];
  }

  x_loc[1] = x_loc[0] + data[0];

  x_loc[2] = x_loc[0] + data[1]*n1dotn2;

  if (z_loc[0]<0)
  {
    y_loc[0] = -y_loc[0];
    y_loc[1] = y_loc[0] - data[1]*n2dotn2prime;
    z_sign = -1;
  }
  else
  {
    y_loc[1] = y_loc[0] + data[1]*n2dotn2prime;
    z_sign = 1;
  }

  z_loc[0] = FABS(z_loc[0]);

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

  CU_TYPE prefac = 1./(4.*M_PI*M_EPS0);

  if (z_loc[0]>1.e-14)
  {
    local_field[0] = z_sign*prefac*ET_Local_Ex(a_loc[0],a_loc[1],
						b_loc[0],b_loc[1],
						u_loc[0],u_loc[1]);
    local_field[1] = prefac*ET_Local_Ey(a_loc[0],a_loc[1],
					 b_loc[0],b_loc[1],
					 u_loc[0],u_loc[1]);
    local_field[2] = prefac*ET_Local_Ez(a_loc[0],a_loc[1],
					 b_loc[0],b_loc[1],
					 u_loc[0],u_loc[1]);
  }
  else
  {
    local_field[0] = z_sign*prefac*ET_Local_Ex(a_loc[0],a_loc[1],
						b_loc[0],b_loc[1],
						u_loc[0],u_loc[1]);
    local_field[1] = prefac*ET_Local_Ey(a_loc[0],a_loc[1],
					 b_loc[0],b_loc[1],
					 u_loc[0],u_loc[1]);
    CU_TYPE p_tmp[3];
    Tri_Centroid(p_tmp,data);
    CU_TYPE dist = SQRT((P[0]-p_tmp[0])*(P[0]-p_tmp[0]) +
			(P[1]-p_tmp[1])*(P[1]-p_tmp[1]) +
			(P[2]-p_tmp[2])*(P[2]-p_tmp[2]));

    if (dist<1.e-12)
      local_field[2] = 1./(2.*M_EPS0);
    else
      local_field[2] = 0.;
  }

  for (int j=0;j<3;j++)
  {
    field_[j] = (data[5 + j]*local_field[0] +
		 N2prime[j]*local_field[1] +
		 norm[j]*local_field[2]);
  }

  CU_TYPE4 field;
  field.x=field_[0];
  field.y=field_[1];
  field.z=field_[2];
  field.w=0.;
  return field;
}

#endif /* KEMFIELD_ELECTROSTATICTRIANGLE_CUH */
