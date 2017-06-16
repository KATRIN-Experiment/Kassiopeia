/*
 * KStaticElectromagnetFieldBuilder.hh
 *
 *  Created on: 26 Mar 2016
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_MAGNETIC_INCLUDE_KSTATICELECTROMAGNETFIELDBUILDER_HH_
#define KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_MAGNETIC_INCLUDE_KSTATICELECTROMAGNETFIELDBUILDER_HH_

#include "KComplexElement.hh"
#include "KStaticElectromagnetFieldWithKGeoBag.hh"
#include "KEMBindingsMessage.hh"
#include "KSmartPointerRelease.hh"

namespace katrin {

typedef KComplexElement< KEMField::KStaticElectromagnetFieldWithKGeoBag > KStaticElectromagnetFieldBuilder;

    template< >
    inline bool KStaticElectromagnetFieldBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            std::string name;
            aContainer->CopyTo(name);
            SetName(name);
            fObject->SetName(name);
            return true;
        }
        if( aContainer->GetName() == "directory" )
        {
            aContainer->CopyTo( fObject, &KEMField::KStaticElectromagnetField::SetDirectory );
            return true;
        }
        if( aContainer->GetName() == "file" )
        {
            aContainer->CopyTo( fObject, &KEMField::KStaticElectromagnetField::SetFile );
            return true;
        }
        if( aContainer->GetName() == "surfaces" )
        {
            std::vector< KGeoBag::KGSurface* > tSurfaces
            = KGeoBag::KGInterface::GetInstance()->RetrieveSurfaces( aContainer->AsReference< std::string >() );

            std::vector< KGeoBag::KGSurface* >::const_iterator tSurfaceIt;
            KGeoBag::KGSurface* tSurface;

            if( tSurfaces.size() == 0 )
            {
                BINDINGMSG( eWarning ) << "no surfaces found for specifier <" << aContainer->AsReference< std::string >() << ">" << eom;
                return false;
            }

            for( tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++ )
            {
                tSurface = *tSurfaceIt;
                fObject->AddSurface( tSurface );
            }
            return true;
        }
        if( aContainer->GetName() == "spaces" )
        {
            std::vector< KGeoBag::KGSpace* > tSpaces =
            KGeoBag::KGInterface::GetInstance()->RetrieveSpaces( aContainer->AsReference< std::string >() );
            std::vector< KGeoBag::KGSpace* >::const_iterator tSpaceIt;
            KGeoBag::KGSpace* tSpace;

            if( tSpaces.size() == 0 )
            {
                BINDINGMSG( eWarning ) << "no spaces found for specifier <" << aContainer->AsReference< std::string >() << ">" << eom;
                return false;
            }

            for( tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++ )
            {
                tSpace = *tSpaceIt;
                fObject->AddSpace( tSpace );
            }
            return true;
        }
        if( aContainer->GetName() == "system" )
        {
            KGeoBag::KGSpace* tSpace =
            KGeoBag::KGInterface::GetInstance()->RetrieveSpace( aContainer->AsReference< std::string >() );

            if( tSpace == NULL )
            {
                BINDINGMSG( eWarning ) << "no spaces found for specifier <" << aContainer->AsReference< std::string >() << ">" << eom;
                return false;
            }

            fObject->SetSystem( tSpace );

            return true;
        }
        return false;
    }

    template< >
    inline bool KStaticElectromagnetFieldBuilder::AddElement( KContainer* anElement )
    {
        if(anElement->Is<KEMField::KMagneticFieldSolver>())
        {
            if(!(fObject->GetFieldSolver())) {
                KEMField::KSmartPointer<KEMField::KMagneticFieldSolver> solver =
                        ReleaseToSmartPtr<KEMField::KMagneticFieldSolver>(anElement);
                fObject->SetFieldSolver(solver);
            } else {
                BINDINGMSG( eError ) << "Cannot set more than one magnetic field"
                        " solver for field " << GetName() << "!" << eom;
            }
        }
        else return false;
        return true;
    }

    template< >
    inline bool KStaticElectromagnetFieldBuilder::End() {
        if (!(fObject->GetFieldSolver()))
        {
            BINDINGMSG( eError ) << " No magnetic field solver "
                    "set in field " << GetName() << "!" << eom;
            return false;
        } else return true;
    }

} /* namespace katrin */

#endif /* KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_MAGNETIC_INCLUDE_KSTATICELECTROMAGNETFIELDBUILDER_HH_ */
