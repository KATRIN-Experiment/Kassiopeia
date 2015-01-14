#ifndef KRING_DEF
#define KRING_DEF

#include "KShape.hh"

#include "KSymmetryGroup.hh"

namespace KEMField
{
  class KRing : public KShape
  {
  public:
    friend class KSymmetryGroup<KRing>;

  protected:

    KRing() : fP(0.,0.,0.) {}
    ~KRing() {}

  public:

    static std::string Name() { return "Ring"; }

    void SetValues(const KPosition& p);

    void SetValues(const double& r,
    		   const double& z);

    double Area() const;
    const KPosition Centroid() const { return fP; }

    double DistanceTo(const KPosition& aPoint, KPosition& nearestPoint);

    const KDirection Normal() const;

    void SetR(double d)         { fP[0] = d; }
    void SetZ(double d)         { fP[2] = d; }
    void SetP(const KPosition& p) { fP = p; }

    const double&   GetR() const { return fP[0]; }
    const double&   GetZ() const { return fP[2]; }
    const KPosition&  GetP() const { return fP; }

    void GetR(double& r) const { r = fP[0]; }
    void GetZ(double& z) const { z = fP[2]; }
    void GetP(KPosition& p) const { p = fP; }

  protected:

    KPosition  fP;
  };

  template <typename Stream>
  Stream& operator>>(Stream& s,KRing& r)
  {
    s.PreStreamInAction(r);
    KPosition p;
    s >> p;
    r.SetValues(p);
    s.PostStreamInAction(r);
    return s;
  }

  template <typename Stream>
  Stream& operator<<(Stream& s,const KRing& r)
  {
    s.PreStreamOutAction(r);
    s << r.GetP();
    s.PostStreamOutAction(r);
    return s;
  }
}

#endif /* KRING_DEF */
