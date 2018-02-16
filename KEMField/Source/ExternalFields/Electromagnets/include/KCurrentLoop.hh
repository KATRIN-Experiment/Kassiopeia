#ifndef KCURRENTLOOP_H
#define KCURRENTLOOP_H

#include "KEMThreeVector.hh"

#include "KElectromagnet.hh"

namespace KEMField
{

  /**
   * @class KCurrentLoop
   *
   * @brief A class describing a loop of current.
   *
   * @author T.J. Corona
   */

  class KCurrentLoop : public KElectromagnet
  {
  public:
    KCurrentLoop() : KElectromagnet(),
		     fP(0.,0.,0.),
		     fCurrent(0.) {}
    virtual ~KCurrentLoop() {}

    static std::string Name() { return "CurrentLoop"; }

    void SetValues(const KPosition& p,
		   double current);

    void SetValues(double r,
		   double z,
		   double current);

    void SetCurrent(double current)     { fCurrent = current; }
    void SetR(double r)                 { fP[0] = r; }
    void SetZ(double z)                 { fP[2] = z; }
    void SetP(const KPosition& p)       { fP = p; }

    double           GetCurrent() const { return fCurrent; }
    double           GetR()       const { return fP[0]; }
    double           GetZ()       const { return fP[2]; }
    const KPosition& GetP()       const { return fP; }

    void Accept(KElectromagnetVisitor& visitor);

  protected:
    KPosition fP;
    double fCurrent;
  };

  template <typename Stream>
  Stream& operator>>(Stream& s,KCurrentLoop& c)
  {
    s.PreStreamInAction(c);
    s >> static_cast<KElectromagnet&>(c);
    KPosition p;
    double current;
    s >> p >> current;
    c.SetValues(p,current);
    s.PostStreamInAction(c);
    return s;
  }

  template <typename Stream>
  Stream& operator<<(Stream& s,const KCurrentLoop& c)
  {
    s.PreStreamOutAction(c);
    s << static_cast<const KElectromagnet&>(c);
    s << c.GetP();
    s << c.GetCurrent();
    s.PostStreamOutAction(c);
    return s;
  }

}

#endif /* KCURRENTLOOP */
