#ifndef Kassiopeia_KSComponentGroupBuilder_h_
#define Kassiopeia_KSComponentGroupBuilder_h_

#include "KComplexElement.hh"
#include "KSComponentGroup.h"
#include "KToolbox.h"

namespace katrin
{

//! a set of components treated as a single composite.
//! @example
//! <output_group name="component_group">
//!     <output name="step_final_particle_nested" parent="step" field="final_particle"/>
//!     <output name="step_final_particle_position_nested" parent="step_final_particle" field="position"/>
//!     <output name="step_final_particle_position_z_nested" parent="step_final_particle_position" field="z"/>
//! </output_group>
typedef KComplexElement<Kassiopeia::KSComponentGroup> KSComponentGroupBuilder;

template<> inline bool KSComponentGroupBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        std::string tName = aContainer->AsString();
        fObject->SetName(tName);
        return true;
    }
    return false;
}

template<> inline bool KSComponentGroupBuilder::AddElement(KContainer* aContainer)
{
    if (aContainer->Is<Kassiopeia::KSComponent>() == true) {
        aContainer->ReleaseTo(fObject, &Kassiopeia::KSComponentGroup::AddComponent);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
