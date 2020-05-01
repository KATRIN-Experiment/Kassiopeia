#ifndef KGCIRCLEWIRE_DEF
#define KGCIRCLEWIRE_DEF

#include "KGBoundary.hh"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace KGeoBag
{

class KGCircleWire : public KGBoundary
{
    /*
	 A class describing a wire circle with a flat section profile
	 */
  public:
    KGCircleWire() {}
    KGCircleWire(double r, double diameter, unsigned int nDisc) : fR(r), fDiameter(diameter), fNDisc(nDisc) {}

    ~KGCircleWire() override {}

    static std::string Name()
    {
        return "circle_wire";
    }

    virtual KGCircleWire* Clone() const;

    virtual void Initialize() const {}
    virtual void AreaInitialize() const override
    {
        Initialize();
    }

    bool ContainsPoint(const double* P) const;
    double DistanceTo(const double* P, double* P_in = nullptr, double* P_norm = nullptr) const;

    double GetLength() const;
    double Area() const;
    double Volume() const;

    void SetR(double d)
    {
        fR = d;
    }
    void SetDiameter(double d)
    {
        fDiameter = d;
    }
    void SetNDisc(unsigned int d)
    {
        fNDisc = d;
    }

    double GetR() const
    {
        return fR;
    }
    double GetDiameter() const
    {
        return fDiameter;
    }
    unsigned int GetNDisc() const
    {
        return fNDisc;
    }

  private:
    double fR;
    double fDiameter;
    unsigned int fNDisc;
};
}  // namespace KGeoBag

#endif /* KGWIRECIRCLE_DEF */
