#ifndef Kassiopeia_KSIntSurfaceMultiplication_h_
#define Kassiopeia_KSIntSurfaceMultiplication_h_

#include "KConst.h"
#include "KField.h"
#include "KSSurfaceInteraction.h"

#include <string>

namespace Kassiopeia
{

/*
    *
    *@file KSIntSurfaceMultiplication.hh
    *@class KSIntSurfaceMultiplication
    *@brief
    * Demo surface interaction (this is not intended to represent any actual physical process)
    *
    *@details
    *
    *<b>Revision History:<b>
    *Date Name Brief Description
    *Sun, 23 Aug 2015 04:11:04   J. Barrett (barrettj@mit.edu) First Version
    *
    */


class KSStep;

class KSIntSurfaceMultiplication : public KSComponentTemplate<KSIntSurfaceMultiplication, KSSurfaceInteraction>
{
  public:
    KSIntSurfaceMultiplication();
    KSIntSurfaceMultiplication(const KSIntSurfaceMultiplication& aCopy);
    KSIntSurfaceMultiplication* Clone() const override;
    ~KSIntSurfaceMultiplication() override;

  public:
    void ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aSecondaries) override;

  public:
    void SetEnergyLossFraction(double frac)
    {
        fEnergyLossFraction = frac;
    };

    double GetEnergyLossFraction() const
    {
        return fEnergyLossFraction;
    };

    void SetEnergyRequiredPerParticle(double e)
    {
        fEnergyRequiredPerParticle = e * katrin::KConst::Q();  //electron volts
    }

    double GetEnergyRequiredPerParticle() const
    {
        return fEnergyRequiredPerParticle;
    };

    void SetSide(std::string side_name)
    {
        //default is both sides of the surface execute the interaction
        fSideName = std::string("both");
        fPerformSideCheck = false;

        //top is the side on which the normal points outward
        if (side_name == "top") {
            fSideName = side_name;
            fPerformSideCheck = true;
            fSideSignIsNegative = true;
        }

        //bottom is the side on which the normal std::vector points inward
        if (side_name == "bottom") {
            fSideName = side_name;
            fPerformSideCheck = true;
            fSideSignIsNegative = false;
        }
    }

    std::string GetSide() const
    {
        return fSideName;
    };

  private:
    bool fPerformSideCheck;
    bool fSideSignIsNegative;
    std::string fSideName;
    double fEnergyLossFraction;
    double fEnergyRequiredPerParticle;
};

}  // namespace Kassiopeia

#endif
