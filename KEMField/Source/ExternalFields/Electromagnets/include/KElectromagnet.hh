#ifndef KELECTROMAGNET_H
#define KELECTROMAGNET_H

#include "KEMCoordinateSystem.hh"

namespace KEMField
{
class KElectromagnetVisitor;

class KElectromagnet
{
  public:
    KElectromagnet() : fCoordinateSystem() {}
    virtual ~KElectromagnet() = default;
    static std::string Name()
    {
        return "Electromagnet";
    }

    KEMCoordinateSystem& GetCoordinateSystem()
    {
        return fCoordinateSystem;
    }

    const KEMCoordinateSystem& GetCoordinateSystem() const
    {
        return fCoordinateSystem;
    }

    virtual void Accept(KElectromagnetVisitor& visitor) = 0;

  protected:
    KEMCoordinateSystem fCoordinateSystem;

    template<typename Stream> friend Stream& operator>>(Stream& s, KElectromagnet& m)
    {
        s.PreStreamInAction(m);
        s >> m.fCoordinateSystem;
        s.PostStreamInAction(m);
        return s;
    }

    template<typename Stream> friend Stream& operator<<(Stream& s, const KElectromagnet& m)
    {
        s.PreStreamOutAction(m);
        s << m.fCoordinateSystem;
        s.PostStreamOutAction(m);
        return s;
    }
};

}  // namespace KEMField

#endif /* KELECTROMAGNET */
