#ifndef KELECTROSTATICLINEARFIELD_DEF
#define KELECTROSTATICLINEARFIELD_DEF

#include "KElectrostaticField.hh"
#include "KGCore.hh"

namespace KEMField
{

class KElectrostaticLinearField : public KElectrostaticField
{
  public:
    KElectrostaticLinearField();
    ~KElectrostaticLinearField() override = default;
    ;

    static std::string Name()
    {
        return "ElectrostaticLinearFieldSolver";
    }

  private:
    double PotentialCore(const KPosition& aSamplePoint) const override;
    KFieldVector ElectricFieldCore(const KPosition& aSamplePoint) const override;

  public:
    void SetPotential1(double aPotential);
    double GetPotential1() const;

    void SetPotential2(double aPotential);
    double GetPotential2() const;

    void SetZ1(double aPosition);
    double GetZ1() const;

    void SetZ2(double aPosition);
    double GetZ2() const;

    void SetSurface(KGeoBag::KGSurface* aSurface);
    const KGeoBag::KGSurface* GetSurface() const;


  protected:
    double fU1, fU2;
    double fZ1, fZ2;
    KGeoBag::KGSurface* fSurface;
};

}  // namespace KEMField

#endif /* KELECTROSTATICLINEARFIELD_DEF */
