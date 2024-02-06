#ifndef Kassiopeia_KSIntCalculatorMott_h_
#define Kassiopeia_KSIntCalculatorMott_h_

#include "KField.h"
#include "KSIntCalculator.h"
#include "TF1.h"

/* 
 * KSintCalculatorMott.h
 * 
 * Date: August 22, 2022
 * Author: Junior Peña (juniorpena)
 */

namespace Kassiopeia
{

/* 
 * The xml configuration for using this calculator is as follows: 
 * <calculator_mott theta_min="[lower_bound]" theta_max="[upper_bound]"/>
 * where [lower_bound] is the lower limit, in radians, on the range of allowed scattering 
 * angles and [upper_bound] is the upper limit. A theta_min of 0 is not allowed since the 
 * Mott Cross Section has a singularity there. 
 */
class KSIntCalculatorMott : public KSComponentTemplate<KSIntCalculatorMott, KSIntCalculator>
{
  public:
    KSIntCalculatorMott();
    KSIntCalculatorMott(const KSIntCalculatorMott& aCopy);
    KSIntCalculatorMott* Clone() const override;
    ~KSIntCalculatorMott() override;

  public:
    void CalculateCrossSection(const KSParticle& anInitialParticle, double& aCrossSection) override;
    void ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aSecondaries) override;

  public:
    ;
    K_SET_GET(double, ThetaMin);  // radians
    ;
    K_SET_GET(double, ThetaMax);  // radians

  public:
    /**
            * \brief Returns scattering angle sampled from Mott Differential Cross Seciton for a given 
            * incoming electron energy. Sampling is done using the GetRandom method for ROOT's TF1 class.
            *  
            * \parameter electron's initial kinetic energy
            * 
            * \return Scatter angle
            */
    double GetTheta(const double& anEnergy);

    /**
            * \brief Calculates energy loss after electron scatters using equation (1.51) in:
            * C. Leroy and P.G. Rancoita, Principles of Radiation Interaction in Matter and Detection, 
            * 2nd Edition, World Scientific (Singapore) 2009.
            *  
            * \parameters electron's initial kinetic energy, scattering angle
            * 
            * \return Electron's energy loss
            */
    double GetEnergyLoss(const double& anEnergy, const double& aTheta);
    /**
            * \brief Initializes Mott Differential Cross Section given in:
            * M.J Boschini, C. Consolandi, M. Gervasi, et al., J. Radiat. Phys. Chem. 90 (2013) 36-66. 
            * Also initializes Total Cross Section, where integration was done separately in mathematica
            * and analytical form written in terms of constants given in publication above. The cross sections
            * are initialized as ROOT TF1 objects.
            *  
            */
    void InitializeMDCS(const double E0);
    /**
            * \brief Deinitializes Mott Differential Cross Section and Mott Total Cross Section
            *  
            */
    void DeinitializeMDCS();

  protected:
    TF1* fMDCS;
    TF1* fMTCS;
};

}  // namespace Kassiopeia

#endif
