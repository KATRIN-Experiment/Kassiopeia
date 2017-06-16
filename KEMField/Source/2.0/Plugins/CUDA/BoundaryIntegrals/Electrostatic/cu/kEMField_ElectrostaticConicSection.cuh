#ifndef KEMFIELD_ELECTROSTATICCONICSECTION_CUH
#define KEMFIELD_ELECTROSTATICCONICSECTION_CUH

#include "kEMField_GaussianQuadrature.cuh"

// Conic section geometry definition (as defined by the streamers in KConicSection.hh):
//
// data[0], data[2]: R1, Z1
// data[3], data[5]: R2, Z2

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE K_elliptic( CU_TYPE eta )
{
  CU_TYPE ln4=1.386294361119890;
  CU_TYPE a[10]={9.657359028085625e-2,3.088514627130518e-2,
		 1.493801353268716e-2,8.789801874555064e-3,
		 6.179627446053317e-3,6.847909282624505e-3,
		 9.848929322176893e-3,8.003003980649985e-3,
		 2.296634898396958e-3,1.393087857006646e-4};
  CU_TYPE b[10]={1.249999999999080e-1,7.031249973903835e-2,
		 4.882804190686239e-2,3.737773975862360e-2,
		 3.012484901289893e-2,2.393191332311079e-2,
		 1.553094163197720e-2,5.973904299155429e-3,
		 9.215546349632498e-4,2.970028096655561e-5};
  CU_TYPE etan,suma,sumb;
  suma=sumb=0.;
  etan=eta;
  int n = 0;
#define UnrollLoop(n)				\
  suma=suma+a[n]*etan;				\
  sumb=sumb+b[n]*etan;				\
  etan=etan*eta;
  UnrollLoop(0);
  UnrollLoop(1);
  UnrollLoop(2);
  UnrollLoop(3);
  UnrollLoop(4);
  UnrollLoop(5);
  UnrollLoop(6);
  UnrollLoop(7);
  UnrollLoop(8);
  UnrollLoop(9);
#undef UnrollLoop
  return ln4+suma-(1./2.+sumb)*LOG(FABS(eta)+1.e-19);
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE E_elliptic( CU_TYPE eta )
{
  CU_TYPE c[10]={4.431471805608895e-1,5.680519456755915e-2,
		 2.183181167613048e-2,1.156959574529540e-2,
		 7.595093422559432e-3,7.820404060959554e-3,
		 1.077063503986645e-2,8.638442173604074e-3,
		 2.468503330460722e-3,1.494662175718132e-4};
  CU_TYPE d[10]={2.499999999999017e-1,9.374999972120314e-2,
		 5.859366125553149e-2,4.271789054738309e-2,
		 3.347894366576162e-2,2.614501470031387e-2,
		 1.680402334636338e-2,6.432146586438301e-3,
		 9.898332846225384e-4,3.185919565550157e-5};

  CU_TYPE etan,sumc,sumd;
  sumc=sumd=0.;
  etan=eta;
#define UnrollLoop(n)						\
  sumc=sumc+c[n]*etan;						\
  sumd=sumd+d[n]*etan;						\
  etan=etan*eta;						\
  if(etan<1.e-20) return 1.+sumc-sumd*LOG(FABS(eta)+1.e-19);
  UnrollLoop(0)
    UnrollLoop(1)
    UnrollLoop(2)
    UnrollLoop(3)
    UnrollLoop(4)
    UnrollLoop(5)
    UnrollLoop(6)
    UnrollLoop(7)
    UnrollLoop(8)
    UnrollLoop(9)
#undef UnrollLoop
    return 1.+sumc-sumd*LOG(FABS(eta)+1.e-19);
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE EK_elliptic( CU_TYPE eta )
{
  CU_TYPE k2,EK,k2n,a,b,cn;
  int n;
  k2=1.-eta;
  if(k2>0.8)
    EK=(E_elliptic(eta)-K_elliptic(eta))/k2;
  else
  {
    a=M_PI/2.;
    k2n=1.;
    EK=0.;
    for(n=1;n<=900;n++)
    {
      cn=(2.*n-1)/(2.*n);
      a=a*cn*cn;
      b=-a/(2.*n-1.);
      EK=EK+(b-a)*k2n;
      k2n=k2n*k2;
      if(k2n<1.e-16) break;
    }
  }
  return EK;
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE EC_PotentialFromChargedRing( const CU_TYPE *P, CU_TYPE *par )
{
  CU_TYPE Z = par[2]+P[0]/par[6]*(par[4]-par[2]);
  CU_TYPE R = par[3]+P[0]/par[6]*(par[5]-par[3]);

  CU_TYPE dz = par[0]-Z;
  CU_TYPE dr = par[1]-R;
  CU_TYPE sumr = R+par[1];

  CU_TYPE eta = (dr*dr+dz*dz)/(sumr*sumr+dz*dz);
  CU_TYPE K = K_elliptic(eta);
  CU_TYPE S = SQRT(sumr*sumr+dz*dz);

  return R*K/S;
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE EC_EFieldRFromChargedRing( const CU_TYPE *P, CU_TYPE *par )
{
  CU_TYPE Z = par[2]+P[0]/par[6]*(par[4]-par[2]);
  CU_TYPE R = par[3]+P[0]/par[6]*(par[5]-par[3]);

  CU_TYPE dz = par[0]-Z;
  CU_TYPE dr = par[1]-R;
  CU_TYPE sumr = R+par[1];

  CU_TYPE eta = (dr*dr+dz*dz)/(sumr*sumr+dz*dz);
  CU_TYPE E = E_elliptic(eta);
  CU_TYPE S = SQRT(sumr*sumr+dz*dz);
  CU_TYPE EK = EK_elliptic(eta);

  return R/(S*S*S)*(-2.*R*EK+(par[1]-R)/eta*E);
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE EC_EFieldZFromChargedRing( const CU_TYPE *P, CU_TYPE *par )
{
  CU_TYPE Z = par[2]+P[0]/par[6]*(par[4]-par[2]);
  CU_TYPE R = par[3]+P[0]/par[6]*(par[5]-par[3]);

  CU_TYPE dz = par[0]-Z;
  CU_TYPE dr = par[1]-R;
  CU_TYPE sumr = R+par[1];

  CU_TYPE eta = (dr*dr+dz*dz)/(sumr*sumr+dz*dz);
  CU_TYPE E = E_elliptic(eta);
  CU_TYPE S = SQRT(sumr*sumr+dz*dz);

  return (par[0]-Z)/(S*S*S)*E/eta*R;
}

//______________________________________________________________________________

GaussianQuadrature(EC_PotentialFromChargedRing)
GaussianQuadrature(EC_EFieldRFromChargedRing)
GaussianQuadrature(EC_EFieldZFromChargedRing)

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE EC_Potential( const CU_TYPE* P, const CU_TYPE* data )
{
  CU_TYPE ln4=1.386294361119890;

  // integration parameters
  CU_TYPE par[7]; // par[0]: z  par[4]: zB
                  // par[1]: r  par[5]: rB
                  // par[2]: zA par[6]: L
                  // par[3]: rA

  CU_TYPE za,ra,zb,rb,L,Da,Db,u[2],z,r,D;
  CU_TYPE q,pp,a,b,pmin,pmax,h;
  int n;

  z  = par[0]=P[2];
  r  = par[1]=SQRT(P[0]*P[0]+P[1]*P[1]);
  za = par[2]=data[2];
  ra = par[3]=data[0];
  zb = par[4]=data[5];
  rb = par[5]=data[3];
  L  = par[6]=SQRT((data[0]-data[3])*(data[0]-data[3]) +
		   (data[2]-data[5])*(data[2]-data[5]));

  Da = SQRT((za-z)*(za-z)+(ra-r)*(ra-r));
  Db = SQRT((zb-z)*(zb-z)+(rb-r)*(rb-r));
  D  = FABS(Da+Db-L)/L;

  if(D>=5.e-2)
    q = GaussianQuadrature_EC_PotentialFromChargedRing(0.,L,20,par);
  else if(D>=5.e-3 && D<5.e-2)
    q = GaussianQuadrature_EC_PotentialFromChargedRing(0.,L,100,par);
  else if(D>=5.e-4 && D<5.e-3)
    q = GaussianQuadrature_EC_PotentialFromChargedRing(0.,L,500,par);
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
      for(n=1;n<=35;n++)
      {
  	q = q + GaussianQuadrature_EC_PotentialFromChargedRing(a,b,50,par);
  	if(FABS(a-pmin)/L<1.e-8) break;
  	b=a;
  	a=pmin+(b-pmin)*0.3;
      }
      h = FABS(a-pmin);
      q=q+h*(ln4+LOG(2.*par[1]+1.e-12)+1.-LOG(h))/2.;
    }
    if(pp>0.)
    {
      pmax = pp;
      pmin = 0.;
      if(pp>L)
  	pmax=L;
      b = pmin;
      a = pmax-(pmax-b)*0.3;
      for(n=1;n<=35;n++)
      {
  	q = q + GaussianQuadrature_EC_PotentialFromChargedRing(b,a,50,par);
  	if(FABS(pmax-a)/L<1.e-8) break;
  	b=a;
  	a=pmax-(pmax-b)*0.3;
      }
      h = FABS(pmax-a);
      q=q+h*(ln4+LOG(2.*par[1]+1.e-12)+1.-LOG(h))/2.;
    }
  }

  return 1./(M_PI*M_EPS0)*q;
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE4 EC_EField( const CU_TYPE* P, const CU_TYPE* data )
{
  CU_TYPE ln4=1.386294361119890;

  // integration parameters
  CU_TYPE par[7]; // par[0]: z  par[4]: zB
  // par[1]: r  par[5]: rB
  // par[2]: zA par[6]: L
  // par[3]: rA

  CU_TYPE za,ra,zb,rb,L,Da,Db,u[2],z,r,D;
  CU_TYPE q[2],pp,a,b,pmin,pmax,h;
  int n;

  z  = par[0]=P[2];
  r  = par[1]=SQRT(P[0]*P[0]+P[1]*P[1]);
  za = par[2]=data[2];
  ra = par[3]=data[0];
  zb = par[4]=data[5];
  rb = par[5]=data[3];
  L  = par[6]=SQRT((data[0]-data[3])*(data[0]-data[3]) +
  		   (data[2]-data[5])*(data[2]-data[5]));

  Da = SQRT((za-z)*(za-z)+(ra-r)*(ra-r));
  Db = SQRT((zb-z)*(zb-z)+(rb-r)*(rb-r));
  D  = FABS(Da+Db-L)/L;

  if(D>=5.e-2)
  {
    q[0] = GaussianQuadrature_EC_EFieldRFromChargedRing(0.,L,20,par);
    q[1] = GaussianQuadrature_EC_EFieldZFromChargedRing(0.,L,20,par);
  }
  else if(D>=5.e-3 && D<5.e-2)
  {
    q[0] = GaussianQuadrature_EC_EFieldRFromChargedRing(0.,L,100,par);
    q[1] = GaussianQuadrature_EC_EFieldZFromChargedRing(0.,L,100,par);
  }
  else if(D>=5.e-4 && D<5.e-3)
  {
    q[0] = GaussianQuadrature_EC_EFieldRFromChargedRing(0.,L,500,par);
    q[1] = GaussianQuadrature_EC_EFieldZFromChargedRing(0.,L,500,par);
  }
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
      for(n=1;n<=35;n++)
      {
  	q[0] = q[0]+GaussianQuadrature_EC_EFieldRFromChargedRing(a,b,50,par);
  	q[1] = q[1]+GaussianQuadrature_EC_EFieldZFromChargedRing(a,b,50,par);
  	if(FABS(a-pmin)/L<1.e-8) break;
  	b=a;
  	a=pmin+(b-pmin)*0.3;
      }
      h = FABS(a-pmin);
      q[0]+=h*(ln4+LOG(2.*par[1]+1.e-12)+1.-LOG(h))/2.;
      q[1]+=h*(ln4+LOG(2.*par[1]+1.e-12)+1.-LOG(h))/2.;
    }
    if(pp>0.)
    {
      pmax = pp;
      pmin = 0.;
      if(pp>L)
  	pmax=L;
      b = pmin;
      a = pmax-(pmax-b)*0.3;
      for(n=1;n<=35;n++)
      {
  	q[0] = q[0]+GaussianQuadrature_EC_EFieldRFromChargedRing(b,a,50,par);
  	q[1] = q[1]+GaussianQuadrature_EC_EFieldZFromChargedRing(b,a,50,par);
  	if(FABS(pmax-a)/L<1.e-8) break;
  	b=a;
  	a=pmax-(pmax-b)*0.3;
      }
      h = FABS(pmax-a);
      q[0]+=h*(ln4+LOG(2.*par[1]+1.e-12)+1.-LOG(h))/2.;
      q[1]+=h*(ln4+LOG(2.*par[1]+1.e-12)+1.-LOG(h))/2.;
    }
  }

  CU_TYPE Er = 1./(M_PI*M_EPS0)*q[0];
  CU_TYPE Ez = 1./(M_PI*M_EPS0)*q[1];

  CU_TYPE4 field;
  field.z = Ez;

  if (par[1]<1.e-14)
  {
    field.x = 0.;
    field.y = 0.;
  }
  else
  {
    CU_TYPE cosine = P[0]/par[1];
    CU_TYPE sine = P[1]/par[1];

    field.x = cosine*Er;
    field.y = sine*Er;
  }

  return field;
}


//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE4 EC_EFieldAndPotential(const CU_TYPE* P, const CU_TYPE* data)
{
  CU_TYPE4 field = EL_EField( P, data );
  CU_TYPE phi = EL_Potential( P, data );

  return MAKECU4( field.x, field.y, field.z, phi );
}

#endif /* KEMFIELD_ELECTROSTATICCONICSECTION_CUH */
