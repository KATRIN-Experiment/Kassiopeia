#include "KGCircularWirePins.hh"

#include "KGExtrudedObject.hh"

namespace KGeoBag
{

  KGCircularWirePins* KGCircularWirePins::Clone() const
  {
    KGCircularWirePins* w = new KGCircularWirePins();

    w->fR1 = fR1;
    w->fR2 = fR2;
    w->fNPins = fNPins;
    w->fDiameter = fDiameter;
    w->fRotationAngle = fRotationAngle;
    w->fNDisc = fNDisc;
    w->fNDiscPower = fNDiscPower;

    return w;
  }

  double KGCircularWirePins::GetLength() const
  {
      // TODO
      return 0.;
  }


  double KGCircularWirePins::Area() const
  {
    // TODO
    return 0.;
  }

  double KGCircularWirePins::Volume() const
  {
    // TODO
    return 0.;
  }

  bool KGCircularWirePins::ContainsPoint(const double* P) const
  {
	// TODO
    (void) P;
    return true;
  }

  double KGCircularWirePins::DistanceTo(const double* P,double* P_in,double* P_norm) const
  {
	// TODO
    (void) P;
    (void) P_in;
    (void) P_norm;
    return 0.;
  }

}
