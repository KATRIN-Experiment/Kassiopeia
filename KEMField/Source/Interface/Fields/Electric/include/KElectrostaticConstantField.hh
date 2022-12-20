#ifndef KELECTROSTATICCONSTANTFIELD_DEF
#define KELECTROSTATICCONSTANTFIELD_DEF

#include "KElectrostaticField.hh"

namespace KEMField
{

class KElectrostaticConstantField : public KElectrostaticField
{
  public:
    KElectrostaticConstantField();
    KElectrostaticConstantField(const KFieldVector& field);
    ~KElectrostaticConstantField() override = default;
    ;

    static std::string Name()
    {
        return "ElectrostaticConstantFieldSolver";
    }

  private:
    double PotentialCore(const KPosition& aSamplePoint) const override;
    KFieldVector ElectricFieldCore(const KPosition& aSamplePoint) const override;

  public:
    void SetField(const KFieldVector& aField);
    KFieldVector GetField() const;

    void SetLocation(const KPosition& aLocation);
    KFieldVector GetLocation() const;

    void SetPotentialOffset(const double& aPotential);
    const double& GetPotentialOffset() const;

  protected:
    KFieldVector fFieldVector;
    KFieldVector fLocation;
    double fPotentialOffset;
};

}  // namespace KEMField

#endif /* KELECTROSTATICCONSTANTFIELD_DEF */
