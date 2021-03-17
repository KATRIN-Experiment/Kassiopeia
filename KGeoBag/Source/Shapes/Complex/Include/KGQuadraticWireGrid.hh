#ifndef KGQUADRATICWIREGRID_DEF
#define KGQUADRATICWIREGRID_DEF

#include "KGBoundary.hh"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace KGeoBag
{
class KGQuadraticWireGrid : public KGBoundary
{
    /*
      A class describing a wire grid with a flat section profile
    */
  public:
    KGQuadraticWireGrid() = default;
    KGQuadraticWireGrid(double r, double pitch, double diameter, unsigned int nDiscPerPitch, bool outerCircle) :
        fR(r),
        fPitch(pitch),
        fDiameter(diameter),
        fNDiscPerPitch(nDiscPerPitch),
        fOuterCircle(outerCircle)
    {}

    ~KGQuadraticWireGrid() override = default;

    static std::string Name()
    {
        return "quadratric_wire_grid";
    }

    virtual KGQuadraticWireGrid* Clone() const;

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
    void SetNDiscPerPitch(unsigned int d)
    {
        fNDiscPerPitch = d;
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
    unsigned int GetNDiscPerPitch() const
    {
        return fNDiscPerPitch;
    }
    bool GetOuterCircle() const
    {
        return fOuterCircle;
    }

  private:
    double fR;
    double fPitch;
    double fDiameter;
    unsigned int fNDiscPerPitch;
    bool fOuterCircle;
};
}  // namespace KGeoBag

#endif /* KGQUDRATICRWIREGRID_DEF */
