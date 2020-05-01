#ifndef KELECTROSTATICCONSTANTFIELD_DEF
#define KELECTROSTATICCONSTANTFIELD_DEF

#include "KElectrostaticField.hh"

namespace KEMField
{

class KElectrostaticConstantField : public KElectrostaticField
{
  public:
    KElectrostaticConstantField();
    KElectrostaticConstantField(const KThreeVector& field);
    ~KElectrostaticConstantField() override{};

    static std::string Name()
    {
        return "ElectrostaticConstantFieldSolver";
    }

  private:
    double PotentialCore(const KPosition& aSamplePoint) const override;
    KThreeVector ElectricFieldCore(const KPosition& aSamplePoint) const override;

  public:
    void SetField(KThreeVector aField);
    KThreeVector GetField() const;

    void SetLocation(const KPosition& aLocation);
    KThreeVector GetLocation() const;

    void SetPotentialOffset(const double& aPotential);
    const double& GetPotentialOffset() const;

  protected:
    KThreeVector fField;
    KThreeVector fLocation;
    double fPotentialOffset;
};

}  // namespace KEMField

#endif /* KELECTROSTATICCONSTANTFIELD_DEF */
