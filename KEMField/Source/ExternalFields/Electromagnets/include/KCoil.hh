#ifndef KCOIL_H
#define KCOIL_H

#include "KElectromagnet.hh"

namespace KEMField
{

/**
   * @class KCoil
   *
   * @brief A class describing a cylindrical solid of current.
   *
   * @author T.J. Corona
   */

class KCoil : public KElectromagnet
{
  public:
    KCoil() : KElectromagnet(), fP0(0., 0., 0.), fP1(0., 0., 0.), fCurrent(0.), fIntegrationScale(30) {}
    ~KCoil() override {}

    static std::string Name()
    {
        return "Coil";
    }

    void SetValues(const KPosition& p0, const KPosition& p1, double current, unsigned int integrationScale);

    void SetValues(double r0, double r1, double z0, double z1, double current, unsigned int integrationScale);

    void SetCurrent(double current)
    {
        fCurrent = current;
    }
    void SetR0(double r0)
    {
        fP0[0] = r0;
    }
    void SetZ0(double z0)
    {
        fP0[2] = z0;
    }
    void SetR1(double r1)
    {
        fP1[0] = r1;
    }
    void SetZ1(double z1)
    {
        fP1[2] = z1;
    }
    void SetP0(const KPosition& p0)
    {
        SetZ0(p0[2]);
        SetR0(sqrt(p0[0] * p0[0] + p0[1] * p0[1]));
    }
    void SetP1(const KPosition& p1)
    {
        SetZ1(p1[2]);
        SetR1(sqrt(p1[0] * p1[0] + p1[1] * p1[1]));
    }
    void SetIntegrationScale(unsigned int i)
    {
        fIntegrationScale = i;
    }

    double GetCurrent() const
    {
        return fCurrent;
    }
    double GetCurrentDensity() const
    {
        return fCurrent / fabs(fP1[0] - fP0[0]) / fabs(fP1[2] - fP0[2]);
    }
    double GetR0() const
    {
        return fP0[0];
    }
    double GetZ0() const
    {
        return fP0[2];
    }
    double GetR1() const
    {
        return fP1[0];
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
    unsigned int GetIntegrationScale() const
    {
        return fIntegrationScale;
    }

    void Accept(KElectromagnetVisitor& visitor) override;

  protected:
    KPosition fP0;
    KPosition fP1;
    double fCurrent;
    unsigned int fIntegrationScale;
};

template<typename Stream> Stream& operator>>(Stream& s, KCoil& c)
{
    s.PreStreamInAction(c);
    s >> static_cast<KElectromagnet&>(c);
    KPosition p0, p1;
    double current;
    unsigned int integrationScale;
    s >> p0 >> p1 >> current >> integrationScale;
    c.SetValues(p0, p1, current, integrationScale);
    s.PostStreamInAction(c);
    return s;
}

template<typename Stream> Stream& operator<<(Stream& s, const KCoil& c)
{
    s.PreStreamOutAction(c);
    s << static_cast<const KElectromagnet&>(c);
    s << c.GetP0();
    s << c.GetP1();
    s << c.GetCurrent();
    s << c.GetIntegrationScale();
    s.PostStreamOutAction(c);
    return s;
}

}  // namespace KEMField

#endif /* KCOIL */
