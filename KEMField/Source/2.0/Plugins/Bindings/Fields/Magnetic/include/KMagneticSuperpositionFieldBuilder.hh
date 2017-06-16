/*
 * KMagneticSuperpositionFieldBuilder.hh
 *
 *  Created on: 4 Apr 2016
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_MAGNETIC_INCLUDE_KMAGNETICSUPERPOSITIONFIELDBUILDER_HH_
#define KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_MAGNETIC_INCLUDE_KMAGNETICSUPERPOSITIONFIELDBUILDER_HH_

#include "KComplexElement.hh"
#include "KMagneticSuperpositionField.hh"
#include "KToolbox.h"

namespace katrin {

struct KMagneticSuperpositionFieldEntry
{
    std::string fName;
    double fEnhancement;
};

typedef KComplexElement< KMagneticSuperpositionFieldEntry > KMagneticSuperpositionFieldEntryBuilder;

template< >
inline bool KMagneticSuperpositionFieldEntryBuilder::AddAttribute( KContainer* aContainer )
{
    if( aContainer->GetName() == "name" )
    {
        aContainer->CopyTo( fObject->fName );
        return true;
    }
    if( aContainer->GetName() == "enhancement" )
    {
        aContainer->CopyTo( fObject->fEnhancement );
        return true;
    }
    return false;
}

typedef KComplexElement< KEMField::KMagneticSuperpositionField > KMagneticSuperpositionFieldBuilder;

template< >
inline bool KMagneticSuperpositionFieldBuilder::AddAttribute( KContainer* aContainer )
{
    if( aContainer->GetName() == "name" )
    {
        std::string name;
        aContainer->CopyTo(name);
        this->SetName(name);
        fObject->SetName(name);
    }
    else if( aContainer->GetName() == "use_caching" )
    {
        aContainer->CopyTo( fObject, &KEMField::KMagneticSuperpositionField::SetUseCaching );
    }
    else return false;
    return true;
}

template< >
inline bool KMagneticSuperpositionFieldBuilder::AddElement( KContainer* aContainer )
{
    if( aContainer->Is< KMagneticSuperpositionFieldEntry >() )
    {
        KMagneticSuperpositionFieldEntry* tNewField =
                aContainer->AsPointer< KMagneticSuperpositionFieldEntry >();

        KEMField::KMagneticField* tMagneticField =
                katrin::KToolbox::GetInstance().Get< KEMField::KMagneticField >(
                        tNewField->fName );

        fObject->AddMagneticField( tMagneticField, tNewField->fEnhancement );
        return true;
    }
    return false;
}



} /* namespace katrin */

#endif /* KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_MAGNETIC_INCLUDE_KMAGNETICSUPERPOSITIONFIELDBUILDER_HH_ */
