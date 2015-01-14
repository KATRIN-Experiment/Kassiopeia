#include "KElectrostaticLineSegmentIntegrator.hh"

namespace KEMField
{
/**
 * \image html potentialFromLineSegment.gif
 * Returns the electric potential at a point P (P[0],P[1],P[2]) due to the 
 * collection of wires by computing the following formula for each copy:
 * \f{eqnarray*}{
 * V(\vec{P}) &=& \frac{\lambda}{4 \pi \epsilon_0} \int_{x_1}^{x^2} \frac{dx}{\sqrt{x^2+z^2}} =\\
 * &=& \frac{\lambda}{4 \pi \epsilon_0} \ln\left( \frac{r_1+r_2+L}{r_1+r_2-L} \right),
 * \f}
 * where the coordinates are as described in the above image.
 */
  double KElectrostaticLineSegmentIntegrator::Potential(const KLineSegment* source,const KPosition& P) const
  {
    double L = (source->GetP1()-source->GetP0()).Magnitude();
    double Da = (source->GetP0()-P).Magnitude();
    double Db = (source->GetP1()-P).Magnitude();

    double ln;

    if((Da+Db) > (L+source->GetDiameter()))
      ln = log((Da+Db+L)/(Da+Db-L));
    else
    {
      KDirection u = (source->GetP1()-source->GetP0())/L;
      double p = (P - source->GetP0()).Dot(u);

      if(p<(-source->GetDiameter()*.5) || p>(L+source->GetDiameter()*.5))
	ln=log((Da+Db+L)/(Da+Db-L));
      else
      {
	KPosition p_ = source->GetP0() + p*u;
	double D = (P - p_).Magnitude();

	if(D>=source->GetDiameter()*.5)
	  ln=log((Da+Db+L)/(Da+Db-L));
	else
	{
	  Da = (source->GetP0() - p_).MagnitudeSquared();
	  Db = (source->GetP1() - p_).MagnitudeSquared();
	  Da = sqrt(Da + source->GetDiameter()*source->GetDiameter()*.25);
	  Db = sqrt(Db + source->GetDiameter()*source->GetDiameter()*.25);

	  ln=log((Da+Db+L)/(Da+Db-L));
	}
      }
    }
    return source->GetDiameter()/(4.*KEMConstants::Eps0)*ln;
  }


  KEMThreeVector KElectrostaticLineSegmentIntegrator::ElectricField(const KLineSegment* source,const KPosition& P) const
  {
    double L = (source->GetP1()-source->GetP0()).Magnitude();
    double Da = (source->GetP0()-P).Magnitude();
    double Db = (source->GetP1()-P).Magnitude();
    KDirection u;

    // if we are not far outside of the wire...
    if(!((Da+Db)>(L+source->GetDiameter())))
    {
      u = (source->GetP1()-source->GetP0())/L;
      double p = (P - source->GetP0()).Dot(u);

      // if we are not just outside of the endcaps of the wire...
      if (!(p<(-source->GetDiameter()*.5) || p>(L + source->GetDiameter()*.5)))
      {
	KPosition p_ = source->GetP0() + p*u;
	double D = (P - p_).Magnitude();

	// if we are not just outside the cylindrical surface of the wire...
	if (!(D>=source->GetDiameter()*.5))
	{
	  // we are in the wire
	  Da = (source->GetP0() - p_).Magnitude();
	  Db = (source->GetP1() - p_).Magnitude();
	}
      }
    }

    double denom = (Da*(Da + Db + L)*(Da + Db - L)*Db)*4.*KEMConstants::Eps0;
    denom = -1./denom;

    return (2.*L*(source->GetP1()*Da - source->GetP0()*(Da + Db) + source->GetP0()*Db))*source->GetDiameter()*denom;
  }

    double KElectrostaticLineSegmentIntegrator::Potential(const KSymmetryGroup<KLineSegment>* source, const KPosition& P) const
    {
      double potential = 0.;
      for (KSymmetryGroup<KLineSegment>::ShapeCIt it=source->begin();it!=source->end();++it)
	potential += Potential(*it,P);
      return potential;
    }

    KEMThreeVector KElectrostaticLineSegmentIntegrator::ElectricField(const KSymmetryGroup<KLineSegment>* source, const KPosition& P) const
    {
      KEMThreeVector electricField(0.,0.,0.);
      for (KSymmetryGroup<KLineSegment>::ShapeCIt it=source->begin();it!=source->end();++it)
	electricField += ElectricField(*it,P);
      return electricField;
    }
}
