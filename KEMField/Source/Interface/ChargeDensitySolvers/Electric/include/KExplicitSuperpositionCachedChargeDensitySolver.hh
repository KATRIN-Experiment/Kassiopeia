/*
 * ExplicitSuperpositionChargeDensitySolver.hh
 *
 *  Created on: 27 Jun 2016
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KEXPLICITSUPERPOSITIONCACHEDCHARGEDENSITYSOLVER_HH_
#define KEMFIELD_SOURCE_2_0_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KEXPLICITSUPERPOSITIONCACHEDCHARGEDENSITYSOLVER_HH_

#include "KExplicitSuperpositionSolutionComponent.hh"
#include "KChargeDensitySolver.hh"
#include <vector>
#include <string>

namespace KEMField {

class KExplicitSuperpositionCachedChargeDensitySolver : public KChargeDensitySolver {
public:
    KExplicitSuperpositionCachedChargeDensitySolver();
    virtual ~KExplicitSuperpositionCachedChargeDensitySolver();

    void SetName( std::string s )
    {
        fName = s;
    }

private:
    void InitializeCore( KSurfaceContainer& container );

public:
    void AddSolutionComponent(KExplicitSuperpositionSolutionComponent* component)
    {
        fNames.push_back(component->name);
        fScaleFactors.push_back(component->scale);
        fHashLabels.push_back(component->hash);
    }

private:

    std::string fName;
    std::vector< std::string > fNames;
    std::vector< double > fScaleFactors;
    std::vector< std::string> fHashLabels;
};

} /* namespace KEMField */

#endif /* KEMFIELD_SOURCE_2_0_CHARGEDENSITYSOLVERS_ELECTRIC_INCLUDE_KEXPLICITSUPERPOSITIONCACHEDCHARGEDENSITYSOLVER_HH_ */
