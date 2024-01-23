/*
* KMagneticImportFieldBuilder.hh
*
*  Created on: 10 March 2022
*      Author: wonyong
*/

#ifndef KMagnetostaticIMPORTFIELDBUILDER_HH_
#define KMagnetostaticIMPORTFIELDBUILDER_HH_

#include "KComplexElement.hh"
#include "KMagnetostaticImportField.hh"
#include "KEMStreamableThreeVector.hh"

namespace katrin {

typedef KComplexElement< KEMField::KMagnetostaticImportField > KSFieldMagneticImportBuilder;

template< >
inline bool KSFieldMagneticImportBuilder::AddAttribute( KContainer* aContainer )
{
    if( aContainer->GetName() == "name" )
    {
        std::string name;
        aContainer->CopyTo(name);
        this->SetName(name);
        fObject->SetName(name);
//        std::cout << "Set KMagnetostaticImportField name \n";

    }

    else if( aContainer->GetName() == "XRange" )
    {
        KEMField::KEMStreamableThreeVector vec;
        aContainer->CopyTo(vec);
        fObject->SetXRange(vec.GetThreeVector());
//        std::cout << "Set KMagnetostaticImportField XRange \n";

    }
    else if( aContainer->GetName() == "YRange" )
    {
        KEMField::KEMStreamableThreeVector vec;
        aContainer->CopyTo(vec);
        fObject->SetYRange(vec.GetThreeVector());
//        std::cout << "Set KMagnetostaticImportField YRange \n";

    }
    else if( aContainer->GetName() == "ZRange" )
    {
        KEMField::KEMStreamableThreeVector vec;
        aContainer->CopyTo(vec);
        fObject->SetZRange(vec.GetThreeVector());
//        std::cout << "Set KMagnetostaticImportField ZRange \n";

    }
    else
    {
        return false;
    }
    return true;
}

} /* namespace katrin */

#endif /* KMagnetostaticIMPORTFIELDBUILDER_HH_ */
