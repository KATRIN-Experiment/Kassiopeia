#ifndef KGLINEARWIREGRID_DEF
#define KGLINEARWIREGRID_DEF

#include "KGBoundary.hh"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace KGeoBag
{
class KGLinearWireGrid : public KGBoundary
{
    /*
      A class describing a wire grid with a flat section profile
    */
  public:
    KGLinearWireGrid() = default;
    KGLinearWireGrid(double r, double pitch, double diameter, unsigned int nDisc, double nDiscPower, bool outerCircle) :
        fR(r),
        fPitch(pitch),
        fDiameter(diameter),
        fNDisc(nDisc),
        fNDiscPower(nDiscPower),
        fOuterCircle(outerCircle)
    {}

    ~KGLinearWireGrid() override = default;

    static std::string Name()
    {
        return "linear_wire_grid";
    }

    virtual KGLinearWireGrid* Clone() const;

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
    void SetPitch(double d)
    {
        fPitch = d;
    }
    void SetDiameter(double d)
    {
        fDiameter = d;
    }
    void SetNDisc(unsigned int d)
    {
        fNDisc = d;
    }
    void SetNDiscPower(double d)
    {
        fNDiscPower = d;
    }
    void SetOuterCircle(bool b)
    {
        fOuterCircle = b;
    }

    double GetR() const
    {
        return fR;
    }
    double GetPitch() const
    {
        return fPitch;
    }
    double GetDiameter() const
    {
        return fDiameter;
    }
    unsigned int GetNDisc() const
    {
        return fNDisc;
    }
    double GetNDiscPower() const
    {
        return fNDiscPower;
    }
    bool GetOuterCircle() const
    {
        return fOuterCircle;
    }

  private:
    double fR;
    double fPitch;
    double fDiameter;
    unsigned int fNDisc;
    double fNDiscPower;
    bool fOuterCircle;
};
}  // namespace KGeoBag

#endif /* KGLINEARWIREGRID_DEF */
