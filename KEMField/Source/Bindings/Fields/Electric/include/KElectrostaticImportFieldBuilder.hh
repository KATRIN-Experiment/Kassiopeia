/*
* KElectricImportFieldBuilder.hh
*
*  Created on: 10 March 2022
*      Author: wonyong
*/

#ifndef KElectrostaticIMPORTFIELDBUILDER_HH_
#define KElectrostaticIMPORTFIELDBUILDER_HH_

#include "KComplexElement.hh"
#include "KElectrostaticImportField.hh"
#include "KEMStreamableThreeVector.hh"

namespace katrin {

typedef KComplexElement< KEMField::KElectrostaticImportField > KSFieldElectricImportBuilder;

template< >
inline bool KSFieldElectricImportBuilder::AddAttribute( KContainer* aContainer )
{
    if( aContainer->GetName() == "name" )
    {
        std::string name;
        aContainer->CopyTo(name);
        this->SetName(name);
        fObject->SetName(name);
        //        std::cout << "Set KElectrostaticImportField name \n";

    }
    else if( aContainer->GetName() == "XRange" )
    {
        KEMField::KEMStreamableThreeVector vec;
        aContainer->CopyTo(vec);
        fObject->SetXRange(vec.GetThreeVector());
        //        std::cout << "Set KElectrostaticImportField XRange \n";

    }
    else if( aContainer->GetName() == "YRange" )
    {
        KEMField::KEMStreamableThreeVector vec;
        aContainer->CopyTo(vec);
        fObject->SetYRange(vec.GetThreeVector());
        //        std::cout << "Set KElectrostaticImportField YRange \n";

    }
    else if( aContainer->GetName() == "ZRange" )
    {
        KEMField::KEMStreamableThreeVector vec;
        aContainer->CopyTo(vec);
        fObject->SetZRange(vec.GetThreeVector());
        //        std::cout << "Set KElectrostaticImportField ZRange \n";

    }
    else
    {
        return false;
    }
    return true;
}

} /* namespace katrin */

#endif /* KElectrostaticIMPORTFIELDBUILDER_HH_ */
