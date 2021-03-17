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
    KGCircleWire() = default;
    KGCircleWire(double r, double diameter, unsigned int nDisc) : fR(r), fDiameter(diameter), fNDisc(nDisc) {}

    ~KGCircleWire() override = default;

    static std::string Name()
    {
        return "circle_wire";
    }

    virtual KGCircleWire* Clone() const;

    virtual void Initialize() const {}
    void AreaInitialize() const override
    {
        Initialize();
    }

    static bool ContainsPoint(const double* P);
    static double DistanceTo(const double* P, const double* P_in = nullptr, const double* P_norm = nullptr);

    static double GetLength();
    static double Area();
    static double Volume();

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
