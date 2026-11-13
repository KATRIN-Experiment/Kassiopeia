/*
 * KCachedChargeDensitySolverBuilder.cc
 *
 *  Created on: 17 Jun 2015
 *      Author: wolfgang
 */

#ifndef SRC_KCACHEDCHARGEDENSITYSOLVERBUILDER_CC_
#define SRC_KCACHEDCHARGEDENSITYSOLVERBUILDER_CC_

#include "KCachedChargeDensitySolverBuilder.hh"

#include "KElectrostaticBoundaryFieldBuilder.hh"

using namespace KEMField;

namespace katrin
{

template<> KCachedChargeDensitySolverBuilder::~KComplexElement() = default;

STATICINT sKCachedChargeDensitySolverStructure =
    //! the name of the surface container. It normally is the the combination of
    //! the word SurfaceContainer and underscore and the hash value (see example in the code).
    //! Try this to avoid comparisions of hash values which may differ from machine to machine.
    //! To get the hash values of the surface containers, run the InspectEMFile command on the .kbd file.
    KCachedChargeDensitySolverBuilder::Attribute<std::string>("name") +
    //! the hash value of the geometry and boundary values.
    //! A solution with the same hash values if present is the loaded from the cache directory.
    //! Be aware that the hash may depend on the machine and one might not find
    //! soludions that have been computed on another machine by the hash value.
    KCachedChargeDensitySolverBuilder::Attribute<std::string>("hash");

STATICINT sKElectrostaticBoundaryField =
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KCachedChargeDensitySolver>("cached_bem_solver") +
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KCachedChargeDensitySolver>("cached_charge_density_solver");
}  // namespace katrin


#endif /* SRC_KCACHEDCHARGEDENSITYSOLVERBUILDER_CC_ */
