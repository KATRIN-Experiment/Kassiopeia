/*
 * KCachedChargeDensitySolverBuilder.hh
 *
 *  Created on: 17 Jun 2015
 *      Author: wolfgang
 */

#ifndef KCACHEDCHARGEDENSITYSOLVERBUILDER_HH_
#define KCACHEDCHARGEDENSITYSOLVERBUILDER_HH_

#include "KCachedChargeDensitySolver.hh"
#include "KComplexElement.hh"

namespace katrin
{

//! This solver loads precomputed charge densities from .kbd files.
//! If a name is provided it will load the surface container with this name from any
//! file in the active directory (normally the KEMField cache directory).
//! If only the hash is provided it will scan the KEMField cache for a solution
//! with the same hash.
//!
//! The solution must have been computed before (e.g. with the same geometry and
//! another charge density solver), so that it is available in the cache.
//! @example
//! <cached_charge_density_solver
//!     name="SurfaceContainer_25630838769cd7b820865b2f7c245ef2"
//!     hash="25630838769cd7b820865b2f7c245ef2"
//! />
typedef KComplexElement<KEMField::KCachedChargeDensitySolver> KCachedChargeDensitySolverBuilder;

template<> inline bool KCachedChargeDensitySolverBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        std::string name;
        aContainer->CopyTo(name);
        fObject->SetName(name);
        return true;
    }
    if (aContainer->GetName() == "hash") {
        std::string hash;
        aContainer->CopyTo(hash);
        fObject->SetHash(hash);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif /* KCACHEDCHARGEDENSITYSOLVERBUILDER_HH_ */
