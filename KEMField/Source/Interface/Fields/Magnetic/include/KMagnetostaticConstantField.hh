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
    KMagnetostaticConstantField(const KFieldVector& aField);
    ~KMagnetostaticConstantField() override = default;
    ;

    static std::string Name()
    {
        return "MagnetostaticConstantFieldSolver";
    }

  private:
    KFieldVector MagneticPotentialCore(const KPosition& aSamplePoint) const override;
    KFieldVector MagneticFieldCore(const KPosition& aSamplePoint) const override;
    KGradient MagneticGradientCore(const KPosition& aSamplePoint) const override;

  public:
    void SetField(const KFieldVector& aFieldVector);
    KFieldVector GetField() const;

    void SetLocation(const KPosition& aLocation);
    KFieldVector GetLocation() const;

  private:
    KFieldVector fFieldVector;
    KFieldVector fLocation;
};

}  // namespace KEMField

#endif /* KMAGNETOSTATICCONSTANTFIELD_HH_ */
