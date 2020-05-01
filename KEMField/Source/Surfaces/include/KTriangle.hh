#ifndef KTRIANGLE_DEF
#define KTRIANGLE_DEF

#include "../../../Surfaces/include/KShape.hh"
#include "../../../Surfaces/include/KSymmetryGroup.hh"

namespace KEMField
{
class KTriangle : public KShape
{
    friend class KSymmetryGroup<KTriangle>;

  protected:
    KTriangle() : fA(0.), fB(0.), fP0(0., 0., 0.), fN1(0., 0., 0.), fN2(0., 0., 0.), fN3(0., 0., 0.) {}
    ~KTriangle() override {}

  public:
    static std::string Name()
    {
        return "Triangle";
    }

    void SetValues(const double& a, const double& b, const KPosition& p0, const KDirection& n1, const KDirection& n2);

    void SetValues(const KPosition& p0, const KPosition& p1, const KPosition& p2);

    double Area() const override
    {
        return .5 * fA * fB * fN1.Cross(fN2).Magnitude();
    }
    const KPosition Centroid() const override
    {
        return fP0 + (fA * fN1 + fB * fN2) / 3.;
    }

    double DistanceTo(const KPosition& aPoint, KPosition& nearestPoint) override;

    const KDirection Normal() const override
    {
        return fN3;
    }

    void SetA(double d)
    {
        fA = d;
    }
    void SetB(double d)
    {
        fB = d;
    }
    void SetP0(const KPosition& p)
    {
        fP0 = p;
    }
    void SetN1(const KDirection& d)
    {
        fN1 = d;
        SetN3();
    }
    void SetN2(const KDirection& d)
    {
        fN2 = d;
        SetN3();
    }
    void SetN3()
    {
        fN3 = (fN1.Cross(fN2)).Unit();
    }

    const double& GetA() const
    {
        return fA;
    }
    const double& GetB() const
    {
        return fB;
    }
    const KPosition& GetP0() const
    {
        return fP0;
    }
    const KDirection& GetN1() const
    {
        return fN1;
    }
    const KDirection& GetN2() const
    {
        return fN2;
    }
    const KDirection& GetN3() const
    {
        return fN3;
    }
    const KPosition GetP1() const
    {
        return fP0 + fN1 * fA;
    }
    const KPosition GetP2() const
    {
        return fP0 + fN2 * fB;
    }

    void GetP0(KPosition& p0) const
    {
        p0 = fP0;
    }
    void GetN1(KDirection& n1) const
    {
        n1 = fN1;
    }
    void GetN2(KDirection& n2) const
    {
        n2 = fN2;
    }
    void GetP1(KPosition& p1) const
    {
        p1 = fP0 + fN1 * fA;
    }
    void GetP2(KPosition& p2) const
    {
        p2 = fP0 + fN2 * fB;
    }

  protected:
    double fA;
    double fB;
    KPosition fP0;
    KDirection fN1;
    KDirection fN2;
    KDirection fN3;
};

template<typename Stream> Stream& operator>>(Stream& s, KTriangle& t)
{
    s.PreStreamInAction(t);
    double a, b;
    KPosition p0;
    KDirection n1, n2;
    s >> a >> b >> p0 >> n1 >> n2;
    t.SetValues(a, b, p0, n1, n2);
    s.PostStreamInAction(t);
    return s;
}

template<typename Stream> Stream& operator<<(Stream& s, const KTriangle& t)
{
    s.PreStreamOutAction(t);
    s << t.GetA();
    s << t.GetB();
    s << t.GetP0();
    s << t.GetN1();
    s << t.GetN2();
    s.PostStreamOutAction(t);
    return s;
}

}  // namespace KEMField

#endif /* KTRIANGLE_DEF */
