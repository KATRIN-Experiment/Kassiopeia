#ifndef Kassiopeia_KESSInelasticBetheFano_h_
#define Kassiopeia_KESSInelasticBetheFano_h_

#include "KESSScatteringCalculator.h"
#include "KField.h"
#include "KSInteractionsMessage.h"

#include <map>

using namespace katrin;

namespace Kassiopeia
{
class KESSPhotoAbsorbtion;
class KESSRelaxation;

class KESSInelasticBetheFano :
    public KSComponentTemplate<KESSInelasticBetheFano, KSIntCalculator>,
    public KESSScatteringCalculator

{
  public:
    KESSInelasticBetheFano();
    KESSInelasticBetheFano(const KESSInelasticBetheFano& aCopy);
    KESSInelasticBetheFano* Clone() const override;
    ~KESSInelasticBetheFano() override;

    void ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aQueue) override;

    void CalculateCrossSection(const KSParticle& aParticle, double& aCrossSection) override;

    K_SET_GET(double, BetheFanoDepositedEnergy)

  private:
    std::map<double, double> fInElScMFPMap;

    //!<map contains a dictionary and a std::vector of two std::vectors containing the values
    std::map<double, std::vector<std::vector<double>>> fInElScMap;

    double fRho; /*This is the density of Silicon in g/Ang^3, this is ridiculous*/

    double CalculateEnergyLoss(const double& EKin);

    double CalculateScatteringAngle(const double& EnergyLoss, const double& KineticEnergy);
};
}  // namespace Kassiopeia

#endif  //KESSINELASTICBETHEFANO_H_
