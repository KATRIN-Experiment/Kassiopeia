#ifndef Kassiopeia_KESSPhotoAbsorbtion_h_
#define Kassiopeia_KESSPhotoAbsorbtion_h_


#include "KField.h"
#include "KSParticle.h"

#include <map>

namespace Kassiopeia
{
class KESSPhotoAbsorbtion
{
  public:
    KESSPhotoAbsorbtion();

    ~KESSPhotoAbsorbtion();

    unsigned int IonizeShell(const double& lostEnergy_eV, const KSParticle& aFinalParticle, KSParticleQueue& aQueue);

    double GetBindingEnergy(unsigned int ionizedShell);

    void CreateSecondary(const double secondaryEnergy, const KSParticle& aFinalParticle, KSParticleQueue& aQueue);

    K_SET_GET(double, SiliconBandGap)
    K_SET_GET(double, PhotoDepositedEnergy)

  protected:
    void ReadIonisationPDF(std::string data_filename);

    unsigned int FindIonizedShell(double lostEnergy_eV);

    std::map<double, unsigned int> fIonizationMap;
    std::vector<double> fShellL1;
    std::vector<double> fShellL2;
    std::vector<double> fShellL3;
    std::vector<double> fShellM;
};

}  // namespace Kassiopeia

#endif /* KESSPhotoAbsorbtion_H_ */
