/*
 * KMagnetostaticConstantField.hh
 *
 *  Created on: 23 Mar 2016
 *      Author: wolfgang
 */

#ifndef KMAGNETOSTATICCONSTANTFIELD_HH_
#define KMAGNETOSTATICCONSTANTFIELD_HH_

#include "KMagnetostaticField.hh"

namespace KEMField
{

class KMagnetostaticConstantField : public KEMField::KMagnetostaticField
{
  public:
    KMagnetostaticConstantField();
    KMagnetostaticConstantField(const KThreeVector& aField);
    ~KMagnetostaticConstantField() override{};

    static std::string Name()
    {
        return "MagnetostaticConstantFieldSolver";
    }

  private:
    KThreeVector MagneticPotentialCore(const KPosition& aSamplePoint) const override;
    KThreeVector MagneticFieldCore(const KPosition& aSamplePoint) const override;
    KGradient MagneticGradientCore(const KPosition& aSamplePoint) const override;

  public:
    void SetField(const KThreeVector& aFieldVector);
    KThreeVector GetField() const;

    void SetLocation(const KPosition& aLocation);
    KThreeVector GetLocation() const;

  private:
    KThreeVector fFieldVector;
    KThreeVector fLocation;
};

}  // namespace KEMField

#endif /* KMAGNETOSTATICCONSTANTFIELD_HH_ */
