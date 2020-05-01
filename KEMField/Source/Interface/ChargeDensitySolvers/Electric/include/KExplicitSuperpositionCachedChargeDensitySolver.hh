/*
 * ExplicitSuperpositionChargeDensitySolver.hh
 *
 *  Created on: 27 Jun 2016
 *      Author: wolfgang
 */

#ifndef KEXPLICITSUPERPOSITIONCACHEDCHARGEDENSITYSOLVER_HH_
#define KEXPLICITSUPERPOSITIONCACHEDCHARGEDENSITYSOLVER_HH_

#include "KChargeDensitySolver.hh"
#include "KExplicitSuperpositionSolutionComponent.hh"

#include <string>
#include <vector>

namespace KEMField
{

class KExplicitSuperpositionCachedChargeDensitySolver : public KChargeDensitySolver
{
  public:
    KExplicitSuperpositionCachedChargeDensitySolver();
    ~KExplicitSuperpositionCachedChargeDensitySolver() override;

    void SetName(std::string s)
    {
        fName = s;
    }

  private:
    void InitializeCore(KSurfaceContainer& container) override;

  public:
    void AddSolutionComponent(KExplicitSuperpositionSolutionComponent* component)
    {
        fNames.push_back(component->name);
        fScaleFactors.push_back(component->scale);
        fHashLabels.push_back(component->hash);
    }

  private:
    std::string fName;
    std::vector<std::string> fNames;
    std::vector<double> fScaleFactors;
    std::vector<std::string> fHashLabels;
};

} /* namespace KEMField */

#endif /* KEXPLICITSUPERPOSITIONCACHEDCHARGEDENSITYSOLVER_HH_ */
