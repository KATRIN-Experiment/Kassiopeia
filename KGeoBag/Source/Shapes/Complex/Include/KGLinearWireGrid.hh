#ifndef KGLINEARWIREGRID_DEF
#define KGLINEARWIREGRID_DEF

#include <stddef.h>
#include <vector>
#include <cmath>
#include <string>

#include "KGBoundary.hh"

namespace KGeoBag
{
  class KGLinearWireGrid : public KGBoundary
  {
    /*
      A class describing a wire grid with a flat section profile
    */
  public:
    KGLinearWireGrid() {}
    KGLinearWireGrid(double r,
			   double pitch,
		       double diameter,
		       unsigned int nDisc,
		       double nDiscPower,
		       bool outerCircle) : fR(r),
						 fPitch(pitch),
					     fDiameter(diameter),
					     fNDisc(nDisc),
					     fNDiscPower(nDiscPower),
					     fOuterCircle(outerCircle){}

    virtual ~KGLinearWireGrid() {}

    static std::string Name() { return "linear_wire_grid"; }

    virtual KGLinearWireGrid* Clone() const;

    virtual void Initialize() const {}

    bool ContainsPoint(const double* P) const;
    double DistanceTo(const double* P,
		      double* P_in=NULL,
		      double* P_norm=NULL) const;

    double GetLength() const;
    double Area() const;
    double Volume() const;

    void SetR (double d) { fR = d; }
    void SetPitch (double d) { fPitch = d; }
    void SetDiameter(double d) { fDiameter = d; }
    void SetNDisc(unsigned int d) { fNDisc = d; }
    void SetNDiscPower(double d) { fNDiscPower = d; }
    void SetOuterCircle( bool b ) { fOuterCircle = b; }

    double GetR() const { return fR;}
    double GetPitch() const { return fPitch; }
    double GetDiameter() const { return fDiameter; }
    unsigned int GetNDisc() const { return fNDisc; }
    double GetNDiscPower() const { return fNDiscPower; }
    bool GetOuterCircle() const { return fOuterCircle; }

  private:

    double fR;
    double fPitch;
    double fDiameter;
    unsigned int fNDisc;
    double fNDiscPower;
    bool fOuterCircle;
  };
}

#endif /* KGLINEARWIREGRID_DEF */
