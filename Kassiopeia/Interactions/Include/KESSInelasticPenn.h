#ifndef Kassiopeia_KESSInelasticPenn_h_
#define Kassiopeia_KESSInelasticPenn_h_

#include "KESSScatteringCalculator.h"
#include "KField.h"

#include <map>

using namespace katrin;

namespace Kassiopeia
{
class KESSPhotoAbsorbtion;
class KESSRelaxation;
class KESSScatteringCalculator;

class KESSInelasticPenn :
    public KSComponentTemplate<KESSInelasticPenn, KSIntCalculator>,
    public KESSScatteringCalculator
{
  public:
    KESSInelasticPenn();
    KESSInelasticPenn(const KESSInelasticPenn& aCopy);
    KESSInelasticPenn* Clone() const override;
    ~KESSInelasticPenn() override;

    void ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aQueue) override;

    void CalculateCrossSection(const KSParticle& aParticle, double& aCrossSection) override;

    K_SET_GET(double, PennDepositedEnergy)

  private:
    std::map<double, double> fInElScMFPMap;

    //!<map contains a dictionary and a std::vector of two std::vectors containing the values
    std::map<double, std::vector<std::vector<double>>> fInElScMap;

    double CalculateEnergyLoss(const double& Ekin);

    double CalculateScatteringAngle(const double EnergyLoss, const double aKineticEnergy);
};

}  // namespace Kassiopeia

#endif  //Kassiopeia_KESSInelasticPenn_h_
