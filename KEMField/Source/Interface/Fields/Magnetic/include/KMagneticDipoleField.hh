/*
 * KMagneticDipoleField.hh
 *
 *  Created on: 24 Mar 2016
 *      Author: wolfgang
 */

#ifndef KMAGNETICDIPOLEFIELD_HH_
#define KMAGNETICDIPOLEFIELD_HH_

#include "KMagnetostaticField.hh"

namespace KEMField
{

class KMagneticDipoleField : public KMagnetostaticField
{
  public:
    KMagneticDipoleField();
    ~KMagneticDipoleField() override;

  private:
    KFieldVector MagneticPotentialCore(const KPosition& aSamplePoint) const override;
    KFieldVector MagneticFieldCore(const KPosition& aSamplePoint) const override;
    KGradient MagneticGradientCore(const KPosition& aSamplePoint) const override;

  public:
    void SetLocation(const KPosition& aLocation);
    void SetMoment(const KDirection& aMoment);

  private:
    KPosition fLocation;
    KDirection fMoment;
};

} /* namespace KEMField */

#endif /* KMAGNETICDIPOLEFIELD_HH_ */
