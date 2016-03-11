#ifndef KGCONICALWIREARRAY_DEF
#define KGCONICALWIREARRAY_DEF

#include <stddef.h>
#include <vector>
#include <cmath>
#include <string>

namespace KGeoBag
{
  class KGConicalWireArray
  {
    /*
      A class describing a wire array with a conic section profile
    */
  public:
    KGConicalWireArray() {}
    KGConicalWireArray(double r1,
		       double z1,
		       double r2,
		       double z2,
		       unsigned int nWires,
		       double thetaStart,
		       double diameter,
		       unsigned int nDisc,
		       double nDiscPower) : fR1(r1),
					     fZ1(z1),
					     fR2(r2),
					     fZ2(z2),
					     fNWires(nWires),
					     fThetaStart(thetaStart),
					     fDiameter(diameter),
					     fNDisc(nDisc),
					     fNDiscPower(nDiscPower){}

    virtual ~KGConicalWireArray() {}

    static std::string Name() { return "conical_wire_array"; }

    virtual KGConicalWireArray* Clone() const;

    virtual void Initialize() const {}

    bool ContainsPoint(const double* P) const;
    double DistanceTo(const double* P,
		      double* P_in=NULL,
		      double* P_norm=NULL) const;
    double GetLength() const;
    double Area() const;
    double Volume() const;

    void SetR1(double d) { fR1 = d; }
    void SetZ1(double d) { fZ1 = d; }
    void SetR2(double d) { fR2 = d; }
    void SetZ2(double d) { fZ2 = d; }
    void SetNWires(unsigned int d) { fNWires = d; }
    void SetThetaStart(double d) { fThetaStart = d; }
    void SetDiameter(double d) { fDiameter = d; }
    void SetNDisc(unsigned int d) { fNDisc = d; }
    void SetNDiscPower(unsigned int d) { fNDiscPower = d; }

    double GetR1() const { return fR1; }
    double GetZ1() const { return fZ1; }
    double GetR2() const { return fR2; }
    double GetZ2() const { return fZ2; }
    unsigned int GetNWires() const { return fNWires; }
    double GetThetaStart() const { return fThetaStart; }
    double GetDiameter() const { return fDiameter; }
    unsigned int GetNDisc() const { return fNDisc; }
    double GetNDiscPower() const { return fNDiscPower; }

  private:

    double fR1;
    double fZ1;
    double fR2;
    double fZ2;
    unsigned int fNWires;
    double fThetaStart;
    double fDiameter;
    unsigned int fNDisc;
    double fNDiscPower;
  };
}

#endif /* KGCONICALWIREARRAY_DEF */
