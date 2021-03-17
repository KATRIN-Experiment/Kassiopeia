#ifndef KCONICSECTION_DEF
#define KCONICSECTION_DEF

#include "../../../Surfaces/include/KShape.hh"
#include "../../../Surfaces/include/KSymmetryGroup.hh"

namespace KEMField
{
class KConicSection : public KShape
{
  public:
    friend class KSymmetryGroup<KConicSection>;

  protected:
    KConicSection() : fP0(0., 0., 0.), fP1(0., 0., 0.) {}
    ~KConicSection() override = default;

  public:
    static std::string Name()
    {
        return "ConicSection";
    }

    void SetValues(const KPosition& p0, const KPosition& p1);

    void SetValues(const double& r0, const double& z0, const double& r1, const double& z1);

    double Area() const override;
    const KPosition Centroid() const override
    {
        return (fP0 + fP1) * .5;
    }

    double DistanceTo(const KPosition& aPoint, KPosition& nearestPoint) override;

    const KDirection Normal() const override;

    void SetR0(double d)
    {
        fP0[0] = d;
    }
    void SetZ0(double d)
    {
        fP0[2] = d;
    }
    void SetR1(double d)
    {
        fP1[0] = d;
    }
    void SetZ1(double d)
    {
        fP1[2] = d;
    }
    void SetP0(const KPosition& p)
    {
        fP0 = p;
    }
    void SetP1(const KPosition& p)
    {
        fP1 = p;
    }

    const double& GetR0() const
    {
        return fP0[0];
    }
    const double& GetZ0() const
    {
        return fP0[2];
    }
    const double& GetR1() const
    {
        return fP1[0];
    }
    const double& GetZ1() const
    {
        return fP1[2];
    }
    const KPosition& GetP0() const
    {
        return fP0;
    }
    const KPosition& GetP1() const
    {
        return fP1;
    }

    void GetR0(double& r0) const
    {
        r0 = fP0[0];
    }
    void GetZ0(double& z0) const
    {
        z0 = fP0[2];
    }
    void GetR1(double& r1) const
    {
        r1 = fP1[0];
    }
    void GetZ1(double& z1) const
    {
        z1 = fP1[2];
    }
    void GetP0(KPosition& p) const
    {
        p = fP0;
    }
    void GetP1(KPosition& p) const
    {
        p = fP1;
    }

  protected:
    KPosition fP0;
    KPosition fP1;
};

template<typename Stream> Stream& operator>>(Stream& s, KConicSection& c)
{
    s.PreStreamInAction(c);
    KPosition p0, p1;
    s >> p0 >> p1;
    c.SetValues(p0, p1);
    s.PostStreamInAction(c);
    return s;
}

template<typename Stream> Stream& operator<<(Stream& s, const KConicSection& c)
{
    s.PreStreamOutAction(c);
    s << c.GetP0();
    s << c.GetP1();
    s.PostStreamOutAction(c);
    return s;
}
}  // namespace KEMField

#endif /* KCONICSECTION_DEF */
