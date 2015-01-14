#ifndef KRECTANGLE_DEF
#define KRECTANGLE_DEF

#include "KShape.hh"

#include "KSymmetryGroup.hh"

namespace KEMField
{
  class KRectangle : public KShape
  {
    friend class KSymmetryGroup<KRectangle>;

  protected:

    KRectangle() : fA(0.),
		   fB(0.),
		   fP0(0.,0.,0.),
		   fN1(0.,0.,0.),
		   fN2(0.,0.,0.),
		   fN3(0.,0.,0.) {}
    virtual ~KRectangle() {}

  public:

    static std::string Name() { return "Rectangle"; }

    void SetValues(const double& a,
		   const double& b,
		   const KPosition& p0,
		   const KDirection& n1,
		   const KDirection& n2);

    void SetValues(const KPosition& p0,
		   const KPosition& p1,
		   const KPosition& /*p2*/,
		   const KPosition& p3);

    double Area() const { return fA*fB; }
    const KPosition Centroid() const { return fP0 + fA*fN1*.5 + fB*fN2*.5; }

    double DistanceTo(const KPosition& aPoint, KPosition& nearestPoint);

    const KDirection Normal() const { return fN3; }

    void              SetA(double d)             { fA = d; }
    void              SetB(double d)             { fB = d; }
    void              SetP0(const KPosition& p)  { fP0 = p; }
    void              SetN1(const KDirection& d) { fN1 = d; SetN3(); }
    void              SetN2(const KDirection& d) { fN2 = d; SetN3(); }
    void              SetN3()                    { fN3 = fN1.Cross(fN2); }

    const double&     GetA()                const { return fA; }
    const double&     GetB()                const { return fB; }
    const KPosition&  GetP0()               const { return fP0; }
    const KDirection& GetN1()               const { return fN1; }
    const KDirection& GetN2()               const { return fN2; }
    const KDirection& GetN3()               const { return fN3; }
    const KPosition   GetP1()               const { return fP0 + fN1*fA; }
    const KPosition   GetP2()               const { return fP0+fN1*fA+fN2*fB;}
    const KPosition   GetP3()               const { return fP0 + fN2*fB; }
    void              GetN1(KDirection& n1) const { n1 = fN1; }
    void              GetN2(KDirection& n2) const { n2 = fN2; }
    void              GetP0(KPosition& p0)  const { p0 = fP0; }
    void              GetP1(KPosition& p1)  const { p1 = fP0 + fN1*fA; }
    void              GetP2(KPosition& p2)  const { p2=fP0+fN1*fA+fN2*fA; }
    void              GetP3(KPosition& p3)  const { p3 = fP0 + fN2*fB; }

  protected:

    double   fA;
    double   fB;
    KPosition  fP0;
    KDirection fN1;
    KDirection fN2;
    KDirection fN3;
  };

  template <typename Stream>
  Stream& operator>>(Stream& s,KRectangle& r)
  {
    s.PreStreamInAction(r);
    double a,b;
    KPosition p0;
    KDirection n1,n2;
    s >> a >> b >> p0 >> n1 >> n2;
    r.SetValues(a,b,p0,n1,n2);
    s.PostStreamInAction(r);
    return s;
  }

  template <typename Stream>
  Stream& operator<<(Stream& s,const KRectangle& r)
  {
    s.PreStreamOutAction(r);
    s << r.GetA();
    s << r.GetB();
    s << r.GetP0();
    s << r.GetN1();
    s << r.GetN2();
    s.PostStreamOutAction(r);
    return s;
  }

}

#endif /* KRECTANGLE_DEF */
