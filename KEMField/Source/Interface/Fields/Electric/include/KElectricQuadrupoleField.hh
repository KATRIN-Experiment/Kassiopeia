/*
 * KElectricQuadrupoleField.hh
 *
 *  Created on: 30 Jul 2015
 *      Author: wolfgang
 */

#ifndef KELECTRICQUADRUPOLEFIELD_HH_
#define KELECTRICQUADRUPOLEFIELD_HH_

#include "KElectrostaticField.hh"

namespace KEMField
{

class KElectricQuadrupoleField : public KElectrostaticField
{
  public:
    KElectricQuadrupoleField();
    ~KElectricQuadrupoleField() override;

    void SetLocation(const KPosition& aLocation);
    void SetStrength(const double& aStrength);
    void SetLength(const double& aLength);
    void SetRadius(const double& aRadius);

  private:
    double PotentialCore(const KPosition& aSamplePoint) const override;
    KFieldVector ElectricFieldCore(const KPosition& aSamplePoint) const override;

    KPosition fLocation;
    double fStrength;
    double fLength;
    double fRadius;
    double fCharacteristic;
};

} /* namespace KEMField */

#endif /* KELECTRICQUADRUPOLEFIELD_HH_ */
