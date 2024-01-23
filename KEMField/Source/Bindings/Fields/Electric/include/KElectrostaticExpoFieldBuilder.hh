#ifndef KELECTROSTATICEXPOFIELDBUILDER_HH_
#define KELECTROSTATICEXPOFIELDBUILDER_HH_

#include "KComplexElement.hh"
#include "KElectrostaticExpoField.hh"
#include "KEMStreamableThreeVector.hh"
#include <iostream>

namespace katrin
{

typedef KComplexElement< KEMField::KElectrostaticExpoField >
    KElectrostaticExpoFieldBuilder;

template< >
inline bool KElectrostaticExpoFieldBuilder::AddAttribute( KContainer* aContainer)
{
    if( aContainer->GetName() == "name" )
    {
        std::string name;
        aContainer->CopyTo(name);
        this->SetName(name);
        fObject->SetName(name);
        return true;
    }
    if (aContainer->GetName() == "TKE") {
        aContainer->CopyTo(fObject, &KEMField::KElectrostaticExpoField::SetTKE);
        return true;
    }
    if (aContainer->GetName() == "B0") {
        aContainer->CopyTo(fObject, &KEMField::KElectrostaticExpoField::SetB0);
        return true;
    }
    if (aContainer->GetName() == "lambda") {
        aContainer->CopyTo(fObject, &KEMField::KElectrostaticExpoField::SetLambda);
        return true;
    }
    if (aContainer->GetName() == "Y0") {
        aContainer->CopyTo(fObject, &KEMField::KElectrostaticExpoField::SetY0);
        return true;
    }

//     if(aContainer->GetName() == "Ey")
//    {
//        KEMField::KEMStreamableThreeVector vec;
//        aContainer->CopyTo(vec);
//        fObject->SetEy(vec.GetThreeVector());
//    }
//     if(aContainer->GetName() == "Epar")
//    {
//        KEMField::KEMStreamableThreeVector vec;
//        aContainer->CopyTo(vec);
//        fObject->SetEpar(vec.GetThreeVector());
//    }

    return false;
}

template< >
inline bool KElectrostaticExpoFieldBuilder::AddElement(KContainer* /*aContainer*/)
{
    return false;
}

} //katrin



#endif /* KELECTROSTATICEXPOFIELDBUILDER_HH_ */

