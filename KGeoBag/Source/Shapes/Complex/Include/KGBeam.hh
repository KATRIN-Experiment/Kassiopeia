#ifndef KGBEAM_DEF
#define KGBEAM_DEF

#include <vector>
#include <cmath>

#include "KGBoundary.hh"
#include "KGCoordinateTransform.hh"

namespace KGeoBag
{
  class KGBeam : public KGBoundary
  {
    /*
      A class describing a straight beam-like object.  The ends of the beam do
      not have to be orthogonal to the beam itself (the beam may have oblique
      faces).
    */
  public:
    KGBeam() : f2DTransform(NULL) {}
    KGBeam(int nDiscRad,
	   int nDiscLong) : fNDiscRad(nDiscRad),
			    fNDiscLong(nDiscLong),
			    f2DTransform(NULL) {}

    virtual ~KGBeam();

    static std::string Name() { return "beam"; }

    virtual KGBeam* Clone() const;

    virtual void Initialize() const;

    bool   ContainsPoint(const double* P) const;
    double DistanceTo(const double* P,double* P_in=NULL,double* P_norm=NULL) const;

    void AddStartLine(double p1[3],double p2[3]);
    void AddEndLine(double p1[3],double p2[3]);

    void SetNDiscRad(int i) { fNDiscRad = i; }
    void SetNDiscLong(int i) { fNDiscLong = i; }

    int    GetNDiscRad()  const { return fNDiscRad; }
    int    GetNDiscLong() const { return fNDiscLong; }

    int    GetRadialDiscretization(unsigned int i) const { return fRadialDisc.at(i);}
    int    GetLongitudinalDiscretization() const { return fNDiscLong;}

    static void LinePlaneIntersection(const double p1[3],
				      const double p2[3],
				      const double p[3],
				      const double n[3],
				      double p_int[3]);

    const std::vector<std::vector<double> >& GetStartCoords() const { return fStartCoords;};
    const std::vector<std::vector<double> >& GetEndCoords() const { return fEndCoords; };


    static double DistanceToLine(const double *P,
				 const double *P1,
				 const double*P2,
				 double *P_in=NULL);

  private:

    void SetRadialDiscretization() const;

    int    fNDiscRad; // Number of discretizations in the radial direction
    int    fNDiscLong; // Number of discretizations along the beam

    std::vector<std::vector<double> > fStartCoords;
    std::vector<std::vector<double> > fEndCoords;

    mutable std::vector<unsigned int> fRadialDisc;

    mutable std::vector<std::vector<double> > f2DCoords;

    // transform to convert to plane 1 coordinates (x and y)
    mutable KGCoordinateTransform* f2DTransform;

    // unit vector pointing from the start coordinates to the end coordinates
    mutable double fUnit[3];

    // unit vector normal to the start coordinate plane
    mutable double fPlane1Norm[3];
    // unit vector normal to the end coordinate plane
    mutable double fPlane2Norm[3];

  };
}

#endif /* KGBEAMOBJECT_DEF */
