/*
 * KChargeDensitySolver.hh
 *
 *  Created on: 01.06.2015
 *      Author: gosda
 */

#ifndef KEMFIELD_SOURCE_2_0_CHARGEDENSITYSOLVERS_ELECTRIC_KCHARGEDENSITYSOLVER_HH_
#define KEMFIELD_SOURCE_2_0_CHARGEDENSITYSOLVERS_ELECTRIC_KCHARGEDENSITYSOLVER_HH_

#include "KSurfaceContainer.hh"

namespace KEMField{

class KChargeDensitySolver
{
public:
    KChargeDensitySolver() : fInitialized(false) {}
    virtual ~KChargeDensitySolver() {}

    void Initialize( KSurfaceContainer& container) {
        if(!fInitialized) {
            InitializeCore(container);
            fInitialized = true;
        }
    }

    void SetHashProperties( unsigned int maskedBits, double hashThreshold);
protected:
    virtual bool FindSolution(double threshold, KSurfaceContainer& container);
    void SaveSolution(double threshold, KSurfaceContainer& container);

private:
    virtual void InitializeCore(KSurfaceContainer& container) = 0;

    unsigned int fHashMaskedBits;
    double fHashThreshold;
    bool fInitialized;
};

}

#endif /* KEMFIELD_SOURCE_2_0_CHARGEDENSITYSOLVERS_ELECTRIC_KCHARGEDENSITYSOLVER_HH_ */
