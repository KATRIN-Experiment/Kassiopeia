/*
* KMagnetostaticExpoFieldBuilder.hh
*
*  Created on: 8 Nov 2017
*      Author: A. Cocco
*/

#ifndef KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_MAGNETIC_INCLUDE_KMAGNETOSTATICEXPOFIELDBUILDER_HH_
#define KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_MAGNETIC_INCLUDE_KMAGNETOSTATICEXPOFIELDBUILDER_HH_

#include "KComplexElement.hh"
#include "KMagnetostaticExpoField.hh"
#include "KEMStreamableThreeVector.hh"

namespace katrin {

typedef KComplexElement< KEMField::KMagnetostaticExpoField > KSFieldMagneticExpoBuilder;

template< >
inline bool KSFieldMagneticExpoBuilder::AddAttribute( KContainer* aContainer )
{
   if( aContainer->GetName() == "name" )
   {
       std::string name;
       aContainer->CopyTo(name);
       this->SetName(name);
       fObject->SetName(name);
       return true;
   }

   if (aContainer->GetName() == "B0") {
       aContainer->CopyTo(fObject, &KEMField::KMagnetostaticExpoField::SetB0);
       return true;
   }
   if (aContainer->GetName() == "lambda") {
       aContainer->CopyTo(fObject, &KEMField::KMagnetostaticExpoField::SetLambda);
       return true;
   }


//   else if( aContainer->GetName() == "Bx" )
//   {
//       KEMField::KEMStreamableThreeVector vec;
//       aContainer->CopyTo(vec);
//       fObject->SetBx(vec.GetThreeVector());
//   }

   return false;
}

} /* namespace katrin */

#endif /* KEMFIELD_SOURCE_2_0_PLUGINS_BINDINGS_FIELDS_MAGNETIC_INCLUDE_KMAGNETOSTATICEXPOFIELDBUILDER_HH_ */
