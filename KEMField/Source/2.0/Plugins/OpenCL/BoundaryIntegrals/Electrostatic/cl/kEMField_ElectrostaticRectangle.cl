#ifndef KEMFIELD_ELECTROSTATICRECTANGLE_CL
#define KEMFIELD_ELECTROSTATICRECTANGLE_CL

#include "kEMField_Rectangle.cl"

// Rectangle geometry definition (as defined by the streamers in KRectangle.hh):
//
// data[0]:     A
// data[1]:     B
// data[2..4]:  P0[0..2]
// data[5..7]:  N1[0..2]
// data[8..10]: N2[0..2]

//______________________________________________________________________________

CL_TYPE ER_Integral_ln(CL_TYPE x,CL_TYPE y,CL_TYPE w)
{
  CL_TYPE r,r0,ret,xa,c1,c2,c3;
  r=SQRT(fabs(x*x+y*y+w*w));
  r0=SQRT(fabs(y*y+w*w));
  xa=fabs(x);

  if(xa<1.e-10)
    c1=0.;
  else
  {
    c1=fabs(y+r)+1.e-12;
    c1 = xa*LOG(c1);
  }
  if(fabs(y)<1.e-12)
    c2=0.;
  else
  {
    c2=fabs((xa+r)/r0)+1.e-12;
    c2 = y*LOG(c2);
  }
  if(fabs(w)<1.e-12)
    c3=0.;
  else
    c3=w*(atan(xa/w)+atan(y*w/(x*x+w*w+xa*r))-atan(y/w));
  ret=c1+c2-xa+c3;
  if(x<0.)
    ret=-ret;
  return ret;
}

//______________________________________________________________________________

CL_TYPE ER_EFieldLocalXY(CL_TYPE x1,
			 CL_TYPE x2,
			 CL_TYPE y1,
			 CL_TYPE y2,
			 CL_TYPE z)
{
  // Computes the x (or y) component of the electric field in local coordinates
  // (where the rectangle lies in the x-y plane, and the field point lies on
  // the z-axis).

  CL_TYPE a1 = y2 + SQRT(y2*y2 + x1*x1 + z*z);
  CL_TYPE a2 = y2 + SQRT(y2*y2 + x2*x2 + z*z);
  CL_TYPE a3 = y1 + SQRT(y1*y1 + x1*x1 + z*z);
  CL_TYPE a4 = y1 + SQRT(y1*y1 + x2*x2 + z*z);

  if (fabs(z)<1.e-14)
  {
    if (fabs(x1)<1.e-14)
    {
      a1 = fabs(y1);
      a3 = fabs(y2);
    }
    if (fabs(x2)<1.e-14)
    {
      a2 = fabs(y1);
      a4 = fabs(y2);
    }
  }

  if (fabs(a1-a3)<1.e-14)
    a1 = a3 = 1.;

  if (fabs(a2-a4)<1.e-14)
    a2 = a4 = 1.;

  return LOG((a2*a3)/(a1*a4));
}

//______________________________________________________________________________

CL_TYPE ER_EFieldLocalZ(CL_TYPE x1,
			CL_TYPE x2,
			CL_TYPE y1,
			CL_TYPE y2,
			CL_TYPE z)
{
  // Computes the z component of the electric field in local coordinates (where
  // the rectangle lies in the x-y plane, and the field point lies on the
  // z-axis).

  CL_TYPE t1 = (fabs(y1)>1.e-15 ? z/y1:1.e15);
  CL_TYPE t2 = (fabs(y2)>1.e-15 ? z/y2:1.e15);

  CL_TYPE g1 = SQRT(((x2*x2+z*z)*t1*t1+z*z)/(x2*x2));
  CL_TYPE g2 = SQRT(((x1*x1+z*z)*t1*t1+z*z)/(x1*x1));
  CL_TYPE g3 = SQRT(((x2*x2+z*z)*t2*t2+z*z)/(x2*x2));
  CL_TYPE g4 = SQRT(((x1*x1+z*z)*t2*t2+z*z)/(x1*x1));

  if (x2<0) g1=-g1;
  if (x1<0) g2=-g2;
  if (x2<0) g3=-g3;
  if (x1<0) g4=-g4;

  return atan(g3)-atan(g4)-atan(g1)+atan(g2);

}

//______________________________________________________________________________

CL_TYPE ER_Potential(const CL_TYPE* P,
		     __global const CL_TYPE* data)
{
  // Rectangle calculation

  CL_TYPE uP = 0;
  CL_TYPE vP = 0;
  CL_TYPE w  = 0;

  CL_TYPE p[3];

  CL_TYPE norm[3];
  Rect_Normal(norm,data);

  int j;
  for (j=0;j<3;j++)
  {
    p[j] = P[j]-data[2+j];
    uP  += p[j]*data[5+j];
    vP  += p[j]*data[8+j];
    w   += p[j]*norm[j];
  }

  CL_TYPE xmin,xmax,ymin,ymax;
  xmin = -uP;
  xmax = -uP + data[0];
  ymin = -vP;
  ymax = -vP + data[1];

  CL_TYPE I = (ER_Integral_ln(xmax,ymax,w) -
	       ER_Integral_ln(xmin,ymax,w) -
	       ER_Integral_ln(xmax,ymin,w) +
	       ER_Integral_ln(xmin,ymin,w));

  return I/(4.*M_PI*M_EPS0);
}

//______________________________________________________________________________

CL_TYPE4 ER_EField(const CL_TYPE* P,
		   __global const CL_TYPE* data)
{
  // Rectangle calculation

  CL_TYPE uP = 0;
  CL_TYPE vP = 0;
  CL_TYPE w  = 0;

  CL_TYPE p[3];

  CL_TYPE norm[3];
  Rect_Normal(norm,data);

  int j;
  for (j=0;j<3;j++)
  {
    p[j] = P[j]-data[2+j];
    uP  += p[j]*data[5+j];
    vP  += p[j]*data[8+j];
    w   += p[j]*norm[j];
  }

  CL_TYPE xmin,xmax,ymin,ymax;
  xmin = -uP;
  xmax = -uP + data[0];
  ymin = -vP;
  ymax = -vP + data[1];

  CL_TYPE prefac = 1./(4.*M_PI*M_EPS0);

  CL_TYPE field_local[3] = {0.,0.,0.};

  field_local[0] = prefac*ER_EFieldLocalXY(xmin,xmax,ymin,ymax,w);
  field_local[1] = prefac*ER_EFieldLocalXY(ymin,ymax,xmin,xmax,w);

  CL_TYPE tmin = w/ymin;
  CL_TYPE tmax = w/ymax;
  CL_TYPE sign_z = 1.;
  if (w<0) sign_z = -1.;
  if (fabs(w)<1.e-13)
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
	fabs(ER_EFieldLocalZ(xmin,xmax,0,fabs(ymin),w) +
	     ER_EFieldLocalZ(xmin,xmax,0,fabs(ymax),w));
    else
      field_local[2] =prefac*sign_z*fabs(ER_EFieldLocalZ(xmin,xmax,ymax,ymin,w));
  }

  CL_TYPE4 field = (CL_TYPE4)(0.,0.,0.,0.);

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

//______________________________________________________________________________

CL_TYPE4 ER_EFieldAndPotential(const CL_TYPE* P,
		    __global const CL_TYPE* data)
{
  CL_TYPE4 field = ER_EField( P, data );
  CL_TYPE phi = ER_Potential( P, data );

  return (CL_TYPE4)( field.s0, field.s1, field.s2, phi );
}

#endif /* KEMFIELD_ELECTROSTATICRECTANGLE_CL */
