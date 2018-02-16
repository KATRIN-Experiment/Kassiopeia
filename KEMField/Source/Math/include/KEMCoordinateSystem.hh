#ifndef KCOORDINATESYSTEM_H
#define KCOORDINATESYSTEM_H

#include "KEMThreeVector.hh"
#include "KEMThreeMatrix.hh"

namespace KEMField{

/**
* @class KEMCoordinateSystem
*
* @brief A class for describing a Cartesian coordinate system.
*
* KEMCoordinateSystem is a class for describing a Cartesian coordinate system
* with respect to a global Cartesian coordinate system, and provides methods for
* transforming between the two systems.
*
* @author T.J. Corona
*/

  class KEMCoordinateSystem
  {
  public:
    KEMCoordinateSystem() : fOrigin(0.,0.,0.),
			    fXAxis(1.,0.,0.),
			    fYAxis(0.,1.,0.),
			    fZAxis(0.,0.,1.) {}
    virtual ~KEMCoordinateSystem() {}

    void SetValues(const KPosition& origin,
		   const KDirection& xAxis,
		   const KDirection& yAxis,
		   const KDirection& zAxis);

    static std::string Name() { return "CoordinateSystem"; }

    KPosition ToLocal(const KPosition& p) const;
    KDirection ToLocal(const KDirection& d) const;
    KEMThreeVector ToLocal(const KEMThreeVector& v) const;
    KGradient ToLocal(const KGradient& g) const;

    KPosition ToGlobal(const KPosition& p) const;
    KDirection ToGlobal(const KDirection& d) const;
    KEMThreeVector ToGlobal(const KEMThreeVector& v) const;
    KGradient ToGlobal(const KGradient& g) const;

    void SetOrigin(const KPosition& origin) { fOrigin = origin; }
    void SetXAxis(const KDirection& xAxis)  { fXAxis = xAxis; }
    void SetYAxis(const KDirection& yAxis)  { fYAxis = yAxis; }
    void SetZAxis(const KDirection& zAxis)  { fZAxis = zAxis; }

    const KPosition&  GetOrigin() const { return fOrigin; }
    const KDirection& GetXAxis()  const { return fXAxis; }
    const KDirection& GetYAxis()  const { return fYAxis; }
    const KDirection& GetZAxis()  const { return fZAxis; }

  protected:
    KPosition fOrigin;
    KDirection fXAxis;
    KDirection fYAxis;
    KDirection fZAxis;
  };

  template <typename Stream>
  Stream& operator>>(Stream& s,KEMCoordinateSystem& c)
  {
    s.PreStreamInAction(c);
    KPosition origin;
    KDirection x,y,z;
    s >> origin >> x >> y >> z;
    c.SetValues(origin,x,y,z);
    s.PostStreamInAction(c);
    return s;
  }

  template <typename Stream>
  Stream& operator<<(Stream& s,const KEMCoordinateSystem& c)
  {
    s.PreStreamOutAction(c);
    s << c.GetOrigin();
    s << c.GetXAxis();
    s << c.GetYAxis();
    s << c.GetZAxis();
    s.PostStreamOutAction(c);
    return s;
  }

  extern KEMCoordinateSystem gGlobalCoordinateSystem;

}

#endif /* KCOORDINATESYSTEM */
