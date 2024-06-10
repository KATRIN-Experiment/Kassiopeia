#ifndef Kassiopeia_KSIntCalculatorMott_h_
#define Kassiopeia_KSIntCalculatorMott_h_

#include "KField.h"
#include "KSIntCalculator.h"

/* 
 * KSintCalculatorMott.h
 * 
 * Date: August 22, 2022
 * Author: Junior Ivan Peña (juniorpe)
 */

namespace Kassiopeia
{

/* 
 * The xml configuration for using this calculator is as follows: 
 * <calculator_mott theta_min="[lower_bound]" theta_max="[upper_bound]"/>
 * where [lower_bound] is the lower limit, in degrees, on the range of allowed scattering 
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
    ;
    K_SET_GET(std::string, Nucleus);

  public:
    /**
            * \brief Returns scattering angle sampled from Mott Differential Cross Seciton for a given 
            * incoming electron energy. Sampling is done using using inverse transform sampling of 
            * Rutherford Differential Cross Section, then rejection sampling with a rescaled Ruth. Diff. X-Sec
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
            * \parameter Electron's initial kinetic energy
            *  
            * \return Electron's natural velocity beta = v/c
            */
    double Beta(double const E0);
    /**
            * \brief Takes in the electron's initial energy and calculates the coefficients used in the 
            * RMott calculation for He corresponding to equation (25) in: 
            * M.J Boschini, C. Consolandi, M. Gervasi, et al., J. Radiat. Phys. Chem. 90 (2013) 36-66.
            * 
            * \parameter Electron's initial kinetic energy
            * 
            * \return Vector containing 5 calculated coeffecients of a
            */
    std::vector<double> RMott_coeffs(double const E0);
    /**
            * \brief Returns the Mott Differential Cross Section(from equation (3) and (24) in: M.J Boschini, 
            * C. Consolandi, M. Gervasi, et al., J. Radiat. Phys. Chem. 90 (2013) 36-66.) Used for rejection 
            * sampling in GetTheta function
            * 
            * \parameter Electron's initial kinetic energy and scatter angle
            *
            * \return Mott Differential Cross Section 
            *  
            */
    double MDCS(double theta, const double E0);
    /**
            * \brief Calculates the analytical integral result of the Mott Differential Cross Section(from
            * equation (3) and (24) in: M.J Boschini, C. Consolandi, M. Gervasi, et al., J. Radiat. Phys. Chem. 90 (2013) 36-66.)
            * within the bounds [fThetaMin, theta] for a given electron initial kinetic energy and some 
            * integral upper bound theta
            * 
            * \parameter Electron's initial kinetic energy, integral upper bound theta
            *
            * \return Integral of Mott Differential Cross Section from fThetaMin to theta
            *  
            */
    double MTCS(double theta, const double E0);
    /**
            * \brief Used for rejection sampling in GetTheta function
            * 
            * \parameter Scatter Angle
            *
            * \return Rutherford Differential Cross Section normalized in range [fThetaMin, fThetaMax] 
            *  
            */
    double Normalized_RDCS(double theta);
    /**
            * \brief Used for inverse transform sampling for obtaining scatter angle in GetTheta function
            *
            * \return Inverse of normalized Rutherford Differential Cross Section in range [fThetaMin, fThetaMax] 
            *  
            */
    double Normalized_RTCSInverse(double x);

};

}  // namespace Kassiopeia

#endif
