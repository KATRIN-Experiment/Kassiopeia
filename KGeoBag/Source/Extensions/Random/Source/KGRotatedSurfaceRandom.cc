#include "KGRotatedSurfaceRandom.hh"

namespace KGeoBag
{
  void KGRotatedSurfaceRandom::VisitWrappedSurface(KGRotatedSurface* rotatedSurface)
  {
    KThreeVector random;
    KSmartPointer<KGRotatedObject> rotatedObject = rotatedSurface->GetObject();

    double area = rotatedObject->Area();
    double sample = Uniform(0.,area);
    double reference = 0.;
    for (unsigned int i=0; i<rotatedObject->GetNSegments(); i++)
    {
      reference += rotatedObject->GetSegment(i)->Area();
      if (sample<reference)
      {
	random = Random(rotatedObject->GetSegment(i));
	break;
      }
    }

    SetRandomPoint(random);
    return;
  }

  KThreeVector KGRotatedSurfaceRandom::Random(const KGRotatedObject::Line* line)
  {
    if (const KGRotatedObject::Arc* arc =
	dynamic_cast<const KGRotatedObject::Arc*>(line))
      return Random(arc);

    // Returns a random point on the truncated cone

    KThreeVector random;

    double x;
    double y;
    double theta;
    double r2;

    if (line->GetAlpha()<1.e-6) // cylinder
    {
      theta = Uniform(0.,2.*M_PI);

      random[0] = line->GetP1(1)*cos(theta);
      random[1] = line->GetP1(1)*sin(theta);
      random[2] = Uniform(line->GetP1(0),line->GetP2(0));
    }
    else // conic section
    {
      do
      {
	x  = Uniform(line->GetUnrolledBoundingBox(0),
		     line->GetUnrolledBoundingBox(2));
	y  = Uniform(line->GetUnrolledBoundingBox(1),
		     line->GetUnrolledBoundingBox(3));

	if (fabs(x)<1.e-14)
	  theta = M_PI/2.;
	else
	  theta = atan(fabs(y/x));
	if (x<1.e-14 && y>-1.e-14)
	  theta = M_PI - theta;
	else if (x<1.e-14 && y<1.e-14)
	  theta += M_PI;
	else if (x>-1.e-14 && y<1.e-14)
	  theta = 2.*M_PI - theta;

	r2 = x*x + y*y;
      }
      while ((theta > line->GetTheta())               ||
	     (r2 < line->GetUnrolledRadius1Squared()) ||
	     (r2 > line->GetUnrolledRadius2Squared()));

      random[2] = line->GetZIntercept() + (line->OpensUp() ? 1.:-1.)*sqrt(r2)*cos(line->GetAlpha());

      theta *= M_PI/line->GetAlpha();
      double r;
      if (fabs(line->GetP1(0)-line->GetP2(0))>1.e-14)
      {
    double m = ((line->GetP1(1)-line->GetP2(1))/
			     (line->GetP1(0)-line->GetP2(0)));
    double b = line->GetP1(0) - m*line->GetP1(1);
	r = m*random[2] + b;
      }
      else
	r = sqrt(r2);

      random[0] = r*cos(theta);
      random[1] = r*sin(theta);
    }
    return random;
  }

  KThreeVector KGRotatedSurfaceRandom::Random(const KGRotatedObject::Arc* arc)
  {
    KThreeVector random;

    double z1 = (arc->GetP1(1) < arc->GetP2(1) ? arc->GetP1(1) : arc->GetP2(1));
    double z2 = (arc->GetP1(1) > arc->GetP2(1) ? arc->GetP1(1) : arc->GetP2(1));
    
    double z,r,alpha;
    do
    {
      z = Uniform(z1,z2);
      r = arc->GetRadius(z);
      alpha = Uniform(0.,arc->GetRMax());
    }
    while (alpha > r);

    double theta = Uniform(0.,2.*M_PI);

    random[0] = r*cos(theta);
    random[1] = r*sin(theta);
    random[2] = z;

    return random;
  }
}
