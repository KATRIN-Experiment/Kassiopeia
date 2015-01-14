#ifndef KLINECURRENT_H
#define KLINECURRENT_H

#include "KElectromagnet.hh"

namespace KEMField
{
  class KLineCurrent : public KElectromagnet
  {

  /**
   * @class KLineCurrent
   *
   * @brief A class describing a line of current.
   *
   * @author T.J. Corona
   */

  public:
    KLineCurrent() : KElectromagnet(),
		     fP0(0.,0.,0.),
		     fP1(0.,0.,0.),
		     fCurrent(0.) {}
    virtual ~KLineCurrent() {}

    void SetValues(const KPosition& p0,
		   const KPosition& p1,
		   double current);

    static std::string Name() { return "LineCurrent"; }

    void SetP0(const KPosition& p0)     { fP0 = p0; }
    void SetP1(const KPosition& p1)     { fP1 = p1; }
    void SetCurrent(double current)     { fCurrent = current; }

    const KPosition& GetP0()      const { return fP0; }
    const KPosition& GetP1()      const { return fP1; }
    double           GetCurrent() const { return fCurrent; }

    void Accept(KElectromagnetVisitor& visitor);

  protected:
    KPosition fP0;
    KPosition fP1;
    double fCurrent;
  };

  template <typename Stream>
  Stream& operator>>(Stream& s,KLineCurrent& c)
  {
    s.PreStreamInAction(c);
    s >> static_cast<KElectromagnet&>(c);
    KPosition p0,p1;
    double current;
    s >> p0 >> p1 >> current;
    c.SetValues(p0,p1,current);
    s.PostStreamInAction(c);
    return s;
  }

  template <typename Stream>
  Stream& operator<<(Stream& s,const KLineCurrent& c)
  {
    s.PreStreamOutAction(c);
    s << static_cast<const KElectromagnet&>(c);
    s << c.GetP0();
    s << c.GetP1();
    s << c.GetCurrent();
    s.PostStreamOutAction(c);
    return s;
  }
}

#endif /* KLINECURRENT */
