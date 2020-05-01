#ifndef Kassiopeia_KESSScatteringCalculator_h_
#define Kassiopeia_KESSScatteringCalculator_h_

#include "KSComponentTemplate.h"
#include "KSIntCalculator.h"

#include <map>

namespace Kassiopeia
{
class KSStep;
class KESSPhotoAbsorbtion;
class KESSRelaxation;
class KESSScatteringModule;

class KESSScatteringCalculator
{
  public:
    KESSScatteringCalculator();
    virtual ~KESSScatteringCalculator();

    //***********
    //composition
    //***********

  public:
    void SetIonisationCalculator(KESSPhotoAbsorbtion* aIonisationCalculator);
    const KESSPhotoAbsorbtion* GetIonisationCalculator() const;
    void SetRelaxationCalculator(KESSRelaxation* aRelaxationCalculator);
    const KESSRelaxation* GetRelaxationCalculator() const;


  protected:
    std::string fInteraction;
    KESSPhotoAbsorbtion* fIonisationCalculator;
    KESSRelaxation* fRelaxationCalculator;

    //******
    //action
    //******

  protected:
    void ReadMFP(std::string data_filename, std::map<double, double>& MapForTables);

    void ReadPDF(std::string data_filename, std::map<double, std::vector<std::vector<double>>>& MapForTables);

    double InterpolateLinear(double x, double x0, double x1, double fx0, double fx1);
};

inline double KESSScatteringCalculator::InterpolateLinear(double x, double x0, double x1, double fx0, double fx1)
{
    return fx0 + (fx1 - fx0) * (x - x0) / (x1 - x0);
}

inline void KESSScatteringCalculator::SetIonisationCalculator(KESSPhotoAbsorbtion* aIonisationCalculator)
{
    fIonisationCalculator = aIonisationCalculator;
}
inline void KESSScatteringCalculator::SetRelaxationCalculator(KESSRelaxation* aRelaxationCalculator)
{
    fRelaxationCalculator = aRelaxationCalculator;
}
inline const KESSPhotoAbsorbtion* KESSScatteringCalculator::GetIonisationCalculator() const
{
    return fIonisationCalculator;
}
inline const KESSRelaxation* KESSScatteringCalculator::GetRelaxationCalculator() const
{
    return fRelaxationCalculator;
}
}  // namespace Kassiopeia

#endif /* Kassiopeia_KESSScatteringCalculator_h_ */
