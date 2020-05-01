#ifndef KSOLENOID_H
#define KSOLENOID_H

#include "KElectromagnet.hh"

namespace KEMField
{

/**
   * @class KSolenoid
   *
   * @brief A class describing a cylinder of current.
   *
   * @author T.J. Corona
   */

class KSolenoid : public KElectromagnet
{
  public:
    KSolenoid() : KElectromagnet(), fP0(0., 0., 0.), fP1(0., 0., 0.), fCurrent(0.) {}
    ~KSolenoid() override {}

    static std::string Name()
    {
        return "Solenoid";
    }

    void SetValues(const KPosition& p0, const KPosition& p1, double current);

    void SetValues(double r, double z0, double z1, double current);

    void SetCurrent(double current)
    {
        fCurrent = current;
    }
    void SetR(double r)
    {
        fP0[0] = fP1[0] = r;
    }
    void SetZ0(double z0)
    {
        fP0[2] = z0;
    }
    void SetZ1(double z1)
    {
        fP1[2] = z1;
    }
    void SetP0(const KPosition& p0)
    {
        SetZ0(p0[2]);
        SetR(sqrt(p0[0] * p0[0] + p0[1] * p0[1]));
    }
    void SetP1(const KPosition& p1)
    {
        SetZ1(p1[2]);
        SetR(sqrt(p1[0] * p1[0] + p1[1] * p1[1]));
    }

    double GetCurrent() const
    {
        return fCurrent;
    }
    double GetCurrentDensity() const
    {
        return fCurrent / fabs(fP1[2] - fP0[2]);
    }
    double GetR() const
    {
        return fP0[0];
    }
    double GetZ0() const
    {
        return fP0[2];
    }
    double GetZ1() const
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

    void Accept(KElectromagnetVisitor& visitor) override;

  protected:
    KPosition fP0;
    KPosition fP1;
    double fCurrent;
};

template<typename Stream> Stream& operator>>(Stream& s, KSolenoid& c)
{
    s.PreStreamInAction(c);
    s >> static_cast<KElectromagnet&>(c);
    KPosition p0, p1;
    double current;
    s >> p0 >> p1 >> current;
    c.SetValues(p0, p1, current);
    s.PostStreamInAction(c);
    return s;
}

template<typename Stream> Stream& operator<<(Stream& s, const KSolenoid& c)
{
    s.PreStreamOutAction(c);
    s << static_cast<const KElectromagnet&>(c);
    s << c.GetP0();
    s << c.GetP1();
    s << c.GetCurrent();
    s.PostStreamOutAction(c);
    return s;
}

}  // namespace KEMField

#endif /* KSOLENOID */
