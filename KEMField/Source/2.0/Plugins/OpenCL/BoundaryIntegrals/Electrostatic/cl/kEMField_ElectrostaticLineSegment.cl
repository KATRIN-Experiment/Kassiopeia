#ifndef KEMFIELD_ELECTROSTATICLINESEGMENT_CL
#define KEMFIELD_ELECTROSTATICLINESEGMENT_CL

#include "kEMField_LineSegment.cl"

#define M_PI 3.141592653589793238462643
#define M_EPS0 8.85418782e-12

// Wire geometry definition (as defined by the streamers in KLineSegment.hh):
//
// data[0..2]: P0[0..2]
// data[3..5]: P1[0..2]
// data[6]:    diameter

//______________________________________________________________________________

CL_TYPE EL_Potential(const CL_TYPE* P,
		       __global const CL_TYPE* data)
{
  // Wire calculation

  CL_TYPE length = SQRT((data[0]-data[3])*(data[0]-data[3]) +
			(data[1]-data[4])*(data[1]-data[4]) +
			(data[2]-data[5])*(data[2]-data[5]));

  CL_TYPE Da = SQRT((data[0]-P[0])*(data[0]-P[0]) +
                    (data[1]-P[1])*(data[1]-P[1]) +
		    (data[2]-P[2])*(data[2]-P[2]));

  CL_TYPE Db = SQRT((data[3]-P[0])*(data[3]-P[0]) +
  	            (data[4]-P[1])*(data[4]-P[1]) +
                    (data[5]-P[2])*(data[5]-P[2]));

  CL_TYPE ln = 0.;

  CL_TYPE p_[3];
  CL_TYPE D = 0.;
  CL_TYPE p = 0.;
  CL_TYPE u[3];

  int i;

  if ((Da+Db) > (length+data[6]))
  {
    ln = LOG((Da+Db+length)/(Da+Db-length));
  }
  else
  {
    for (i=0;i<3;i++)
    {
      u[i] = (data[3+i]-data[i])/length;
      p = p + (P[i]-data[i])*u[i];
    }

    if (p<(-data[6]*.5) || p>(length+data[6]*.5))
      ln = LOG((Da+Db+length)/(Da+Db-length));
    else
    {
      for (i=0;i<3;i++)
      {
        p_[i] = data[i] + p*u[i];
        D = D + (P[i]-p_[i])*(P[i]-p_[i]);
      }
      D = SQRT(D);

      if (D>=data[6]*.5)
      {
        ln = LOG((Da+Db+length)/(Da+Db-length));
      }
      else
      {
        Da = 0.;
        Db = 0.;

        for (i=0;i<3;i++)
        {
          Da = Da + (data[i] - p_[i])*(data[i] - p_[i]);
	  Db = Db + (data[3+i] - p_[i])*(data[3+i] - p_[i]);
        }
        Da = SQRT(Da + data[6]*data[6]*.25);
        Db = SQRT(Db + data[6]*data[6]*.25);
        ln = LOG((Da+Db+length)/(Da+Db-length));
      }
    }

  }

  return data[6]/(4.*M_EPS0)*ln;
}

//______________________________________________________________________________

CL_TYPE4 EL_EField(const CL_TYPE* P,
		     __global const CL_TYPE* data)
{
  // Wire calculation

  CL_TYPE length = SQRT((data[0]-data[3])*(data[0]-data[3]) +
			(data[1]-data[4])*(data[1]-data[4]) +
			(data[2]-data[5])*(data[2]-data[5]));

  CL_TYPE Da = SQRT((data[0]-P[0])*(data[0]-P[0]) +
                    (data[1]-P[1])*(data[1]-P[1]) +
                    (data[2]-P[2])*(data[2]-P[2]));

  CL_TYPE Db = SQRT((data[3]-P[0])*(data[3]-P[0]) +
                    (data[4]-P[1])*(data[4]-P[1]) +
                    (data[5]-P[2])*(data[5]-P[2]));

  CL_TYPE p_[3];
  CL_TYPE D = 0.;
  CL_TYPE p = 0.;
  CL_TYPE u[3];
  CL_TYPE f[3];
  CL_TYPE4 field = (CL_TYPE4)(0.,0.,0.,0.);
  CL_TYPE denom;

  int i;

  if ((Da+Db) < (length+data[6]))
  {
    for (i=0;i<3;i++)
    {
      u[i] = (data[3+i]-data[i])/length;
      p = p + (P[i]-data[i])*u[i];
    }

    if (p>(-data[6]*.5) && p<(length+data[6]*.5))
    {
      for (i=0;i<3;i++)
      {
        p_[i] = data[i] + p*u[i];
        D = D + (P[i]-p_[i])*(P[i]-p_[i]);
      }
      D = SQRT(D);

      if (D<data[6]*.5)
      {
        Da = 0.;
        Db = 0.;

        for (i=0;i<3;i++)
        {
          Da = Da + (data[i] - p_[i])*(data[i] - p_[i]);
          Db = Db + (data[3+i] - p_[i])*(data[3+i] - p_[i]);
        }
        Da = SQRT(Da + data[6]*data[6]*.25);
        Db = SQRT(Db + data[6]*data[6]*.25);
      }
    }
  }

  denom = (Da*(Da+Db+length)*(Da+Db-length)*Db);
  denom = 1./denom;

  for (i=0;i<3;i++)
  {
    f[i] = -(2.*length*(data[3+i]*Da-P[i]*(Da+Db)+data[i]*Db))*denom;
  }

  field.x = data[6]/(4.*M_EPS0)*f[0];
  field.y = data[6]/(4.*M_EPS0)*f[1];
  field.z = data[6]/(4.*M_EPS0)*f[2];

  return field;
}

#endif /* KEMFIELD_ELECTROSTATICLINESEGMENT_CL */
