#ifndef KGROD_DEF
#define KGROD_DEF

#include "KGBoundary.hh"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace KGeoBag
{
class KGRod : public KGBoundary
{
    /*
      A class describing a rod.  The rod can be bent into arbitrary shapes, as
      long as each rod segment is a straight line.
    */
  public:
    KGRod() = default;
    KGRod(double radius, int nDiscRad, int nDiscLong) : fRadius(radius), fNDiscRad(nDiscRad), fNDiscLong(nDiscLong) {}

    ~KGRod() override = default;

    static std::string Name()
    {
        return "rod";
    }

    virtual KGRod* Clone() const;

    virtual void Initialize() const {}
    void AreaInitialize() const override
    {
        Initialize();
    }

    bool ContainsPoint(const double* P) const;
    double DistanceTo(const double* P, double* P_in = nullptr, double* P_norm = nullptr) const;
    double Area() const;
    double Volume() const;

    void AddPoint(const double p[3]);

    double GetLength() const;

    void SetRadius(double d)
    {
        fRadius = d;
    }
    void SetNDiscRad(int i)
    {
        fNDiscRad = i;
    }
    void SetNDiscLong(int i)
    {
        fNDiscLong = i;
    }

    double GetRadius() const
    {
        return fRadius;
    }
    int GetNDiscRad() const
    {
        return fNDiscRad;
    }
    int GetNDiscLong() const
    {
        return fNDiscLong;
    }

    unsigned int GetNCoordinates() const
    {
        return fCoords.size();
    }
    const double* GetCoordinate(unsigned int i) const
    {
        return &(fCoords.at(i).at(0));
    }
    double GetCoordinate(unsigned int i, unsigned int j) const
    {
        return fCoords.at(i).at(j);
    }

    static void Normalize(const double* p1, const double* p2, double* norm);

    static void GetNormal(const double* p1, const double* p2, const double* oldNormal, double* normal);

  private:
    std::vector<std::vector<double>> fCoords;

    double fRadius;

    // Number of discretizations in the radial direction
    int fNDiscRad;
    // Number of discretizations along the rod
    int fNDiscLong;
};
}  // namespace KGeoBag

#endif /* KGROD_DEF */
