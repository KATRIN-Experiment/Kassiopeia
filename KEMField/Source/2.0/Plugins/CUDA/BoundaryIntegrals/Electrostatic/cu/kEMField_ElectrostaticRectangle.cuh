#ifndef KEMFIELD_ELECTROSTATICRECTANGLE_CUH
#define KEMFIELD_ELECTROSTATICRECTANGLE_CUH

// Rectangle geometry definition (as defined by the streamers in KRectangle.hh):
//
// data[0]:     A
// data[1]:     B
// data[2..4]:  P0[0..2]
// data[5..7]:  N1[0..2]
// data[8..10]: N2[0..2]

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE ER_Integral_ln( CU_TYPE x, CU_TYPE y, CU_TYPE w )
{
  CU_TYPE r,r0,ret,xa,c1,c2,c3;
  r=SQRT(FABS(x*x+y*y+w*w));
  r0=SQRT(FABS(y*y+w*w));
  xa=FABS(x);

  if(xa<1.e-10)
    c1=0.;
  else
  {
    c1=FABS(y+r)+1.e-12;
    c1 = xa*LOG(c1);
  }
  if(FABS(y)<1.e-12)
    c2=0.;
  else
  {
    c2=FABS((xa+r)/r0)+1.e-12;
    c2 = y*LOG(c2);
  }
  if(FABS(w)<1.e-12)
    c3=0.;
  else
    c3=w*(ATAN(xa/w)+ATAN(y*w/(x*x+w*w+xa*r))-ATAN(y/w));
  ret=c1+c2-xa+c3;
  if(x<0.)
    ret=-ret;
  return ret;
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE ER_EFieldLocalXY( CU_TYPE x1,
                            CU_TYPE x2,
                            CU_TYPE y1,
                            CU_TYPE y2,
                            CU_TYPE z)
{
  // Computes the x (or y) component of the electric field in local coordinates
  // (where the rectangle lies in the x-y plane, and the field point lies on
  // the z-axis).

  CU_TYPE a1 = y2 + SQRT(y2*y2 + x1*x1 + z*z);
  CU_TYPE a2 = y2 + SQRT(y2*y2 + x2*x2 + z*z);
  CU_TYPE a3 = y1 + SQRT(y1*y1 + x1*x1 + z*z);
  CU_TYPE a4 = y1 + SQRT(y1*y1 + x2*x2 + z*z);

  if (FABS(z)<1.e-14)
  {
    if (FABS(x1)<1.e-14)
    {
      a1 = FABS(y1);
      a3 = FABS(y2);
    }
    if (FABS(x2)<1.e-14)
    {
      a2 = FABS(y1);
      a4 = FABS(y2);
    }
  }

  if (FABS(a1-a3)<1.e-14)
    a1 = a3 = 1.;

  if (FABS(a2-a4)<1.e-14)
    a2 = a4 = 1.;

  return LOG((a2*a3)/(a1*a4));
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE ER_EFieldLocalZ(CU_TYPE x1,
                        CU_TYPE x2,
                        CU_TYPE y1,
                        CU_TYPE y2,
                        CU_TYPE z)
{
  // Computes the z component of the electric field in local coordinates (where
  // the rectangle lies in the x-y plane, and the field point lies on the
  // z-axis).

  CU_TYPE t1 = (FABS(y1)>1.e-15 ? z/y1:1.e15);
  CU_TYPE t2 = (FABS(y2)>1.e-15 ? z/y2:1.e15);

  CU_TYPE g1 = SQRT(((x2*x2+z*z)*t1*t1+z*z)/(x2*x2));
  CU_TYPE g2 = SQRT(((x1*x1+z*z)*t1*t1+z*z)/(x1*x1));
  CU_TYPE g3 = SQRT(((x2*x2+z*z)*t2*t2+z*z)/(x2*x2));
  CU_TYPE g4 = SQRT(((x1*x1+z*z)*t2*t2+z*z)/(x1*x1));

  if (x2<0) g1=-g1;
  if (x1<0) g2=-g2;
  if (x2<0) g3=-g3;
  if (x1<0) g4=-g4;

  return ATAN(g3)-ATAN(g4)-ATAN(g1)+ATAN(g2);

}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE ER_Potential( const CU_TYPE* P, const CU_TYPE* data )
{
  // Rectangle calculation

  CU_TYPE uP = 0;
  CU_TYPE vP = 0;
  CU_TYPE w  = 0;

  CU_TYPE p[3];

  CU_TYPE norm[3];
  Rect_Normal(norm,data);

  int j;
  for (j=0;j<3;j++)
  {
    p[j] = P[j]-data[2+j];
    uP  += p[j]*data[5+j];
    vP  += p[j]*data[8+j];
    w   += p[j]*norm[j];
  }

  CU_TYPE xmin,xmax,ymin,ymax;
  xmin = -uP;
  xmax = -uP + data[0];
  ymin = -vP;
  ymax = -vP + data[1];

  CU_TYPE I = (ER_Integral_ln(xmax,ymax,w) -
	       ER_Integral_ln(xmin,ymax,w) -
	       ER_Integral_ln(xmax,ymin,w) +
	       ER_Integral_ln(xmin,ymin,w));

  return I/(4.*M_PI*M_EPS0);
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE4 ER_EField( const CU_TYPE* P, const CU_TYPE* data )
{
  // Rectangle calculation

  CU_TYPE uP = 0;
  CU_TYPE vP = 0;
  CU_TYPE w  = 0;

  CU_TYPE p[3];

  CU_TYPE norm[3];
  Rect_Normal(norm,data);

  int j;
  for (j=0;j<3;j++)
  {
    p[j] = P[j]-data[2+j];
    uP  += p[j]*data[5+j];
    vP  += p[j]*data[8+j];
    w   += p[j]*norm[j];
  }

  CU_TYPE xmin,xmax,ymin,ymax;
  xmin = -uP;
  xmax = -uP + data[0];
  ymin = -vP;
  ymax = -vP + data[1];

  CU_TYPE prefac = 1./(4.*M_PI*M_EPS0);

  CU_TYPE field_local[3] = {0.,0.,0.};

  field_local[0] = prefac*ER_EFieldLocalXY(xmin,xmax,ymin,ymax,w);
  field_local[1] = prefac*ER_EFieldLocalXY(ymin,ymax,xmin,xmax,w);

  CU_TYPE tmin = w/ymin;
  CU_TYPE tmax = w/ymax;
  CU_TYPE sign_z = 1.;
  if (w<0) sign_z = -1.;
  if (FABS(w)<1.e-13)
  {
    if (xmin<0. && xmax>0. && ymin<0. && ymax>0.)
    {
            //changed 12/8/14 JB
            //We are probably on the surface of the element
            //so ignore any z displacement (which may be a round off error)
            //and always pick a consistent direction (aligned with normal vector)
            field_local[2] = 1.0/(2.*M_EPS0);
            //field_local[2] = sign_z/(2.*M_EPS0);
    }
    else
    {
      field_local[2] = 0.;
    }
  }
  else
  {
    if (((tmin>0) - (tmin<0)) != ((tmax>0) - (tmax<0)))
      field_local[2] = prefac*sign_z*
	FABS(ER_EFieldLocalZ(xmin,xmax,0,FABS(ymin),w) +
	     ER_EFieldLocalZ(xmin,xmax,0,FABS(ymax),w));
    else
      field_local[2] =prefac*sign_z*FABS(ER_EFieldLocalZ(xmin,xmax,ymax,ymin,w));
  }

  CU_TYPE4 field;
  field.x=0.; field.y=0.; field.z=0.; field.w=0.;

  field.x = (data[5]*field_local[0] +
  	     data[8]*field_local[1] +
  	     norm[0]*field_local[2]);
  field.y = (data[6]*field_local[0] +
  	     data[9]*field_local[1] +
  	     norm[1]*field_local[2]);
  field.z = (data[7]*field_local[0] +
  	     data[10]*field_local[1] +
  	     norm[2]*field_local[2]);

  return field;
}

#endif /* KEMFIELD_ELECTROSTATICRECTANGLE_CUH */
