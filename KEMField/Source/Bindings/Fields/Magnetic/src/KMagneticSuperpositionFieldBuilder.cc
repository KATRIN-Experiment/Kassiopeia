/*
 * KMagneticSuperpositionFieldBuilder.cc
 *
 *  Created on: 4 Apr 2016
 *      Author: wolfgang
 */

#include "KMagneticSuperpositionFieldBuilder.hh"

#include "KEMToolboxBuilder.hh"

using namespace KEMField;
using namespace std;

namespace katrin
{

STATICINT sKMagneticSuperpositionFieldEntryStructure =
    //! the name of the field that takes part in the superposition.
    KMagneticSuperpositionFieldEntryBuilder::Attribute<std::string>("name") +
    //! the scale factor of the field for the superposition.
    KMagneticSuperpositionFieldEntryBuilder::Attribute<double>("enhancement");


STATICINT sKMagneticSuperpositionFieldStructure =
    //! the name of the superposition field
    KMagneticSuperpositionFieldBuilder::Attribute<std::string>("name") +
    //! store computed values for the magnetic potential, field or gradient
    //! in a cache. The next time the same point will be evaluated, the
    //! value is already available from the cache.
    //! This works only with static fields.
    KMagneticSuperpositionFieldBuilder::Attribute<bool>("use_caching") +
    KMagneticSuperpositionFieldBuilder::Attribute<std::string>("require") +
    //! add a magnetic field to the superposition and specify the factor
    //! by which its intensity is scaled.
    KMagneticSuperpositionFieldBuilder::ComplexElement<KMagneticSuperpositionFieldEntry>("add_field");


STATICINT sKEMFieldToolboxStructure =
    KEMToolboxBuilder::ComplexElement<KMagneticSuperpositionField>("magnetic_superposition_field");

} /* namespace katrin */
