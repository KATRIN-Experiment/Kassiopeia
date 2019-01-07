#ifndef KLINESEGMENT_DEF
#define KLINESEGMENT_DEF

#include "../../../Surfaces/include/KShape.hh"
#include "../../../Surfaces/include/KSymmetryGroup.hh"

namespace KEMField
{
  class KLineSegment : public KShape
  {
    friend class KSymmetryGroup<KLineSegment>;

  protected:

    KLineSegment() : fP0(0.,0.,0.),
		     fP1(0.,0.,0.),
		     fDiameter(0.) {}
    ~KLineSegment() {}

  public:

    static std::string Name() { return "LineSegment"; }

    void SetValues(const KPosition& p0,
		   const KPosition& p1,
		   const double&  diameter);

    double Area() const { return M_PI*fDiameter*(fP0-fP1).Magnitude(); }
    const KPosition Centroid() const { return (fP0 + fP1)*.5; }

    double DistanceTo(const KPosition& aPoint, KPosition& nearestPoint);

    const KDirection Normal() const;

    void             SetP0(const KPosition& p)    { fP0 = p; }
    void             SetP1(const KPosition& p)    { fP1 = p; }
    void             SetDiameter(const double& d) { fDiameter = d; }

    const KPosition& GetP0()                const { return fP0; }
    const KPosition& GetP1()                const { return fP1; }
    const double&    GetDiameter()          const { return fDiameter; }

    void             GetP0(KPosition& p0)   const { p0 = fP0; }
    void             GetP1(KPosition& p1)   const { p1 = fP1; }

  protected:

    KPosition fP0;
    KPosition fP1;
    double  fDiameter;
  };

  template <typename Stream>
  Stream& operator>>(Stream& s,KLineSegment& t)
  {
    s.PreStreamInAction(t);
    KPosition p0,p1;
    double diameter;
    s >> p0 >> p1 >> diameter;
    t.SetValues(p0,p1,diameter);
    s.PostStreamInAction(t);
    return s;
  }

  template <typename Stream>
  Stream& operator<<(Stream& s,const KLineSegment& t)
  {
    s.PreStreamOutAction(t);
    s << t.GetP0();
    s << t.GetP1();
    s << t.GetDiameter();
    s.PostStreamOutAction(t);
    return s;
  }

}

#endif /* KTRIANGLE_DEF */
