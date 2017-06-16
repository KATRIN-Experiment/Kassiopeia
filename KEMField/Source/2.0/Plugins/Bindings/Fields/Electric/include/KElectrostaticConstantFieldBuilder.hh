#ifndef KELECTROSTATICCONSTANTFIELDBUILDER_HH_
#define KELECTROSTATICCONSTANTFIELDBUILDER_HH_

#include "KComplexElement.hh"
#include "KElectrostaticConstantField.hh"
#include "KEMStreamableThreeVector.hh"
#include <iostream>

namespace katrin
{

	typedef KComplexElement< KEMField::KElectrostaticConstantField >
			KElectrostaticConstantFieldBuilder;

	template< >
	inline bool KElectrostaticConstantFieldBuilder::AddAttribute( KContainer* aContainer)
	{
		if( aContainer->GetName() == "name" )
		{
			std::string name;
			aContainer->CopyTo(name);
			this->SetName(name);
			fObject->SetName(name);
		}
		else if(aContainer->GetName() == "field")
		{
			KEMField::KEMStreamableThreeVector vec;
			aContainer->CopyTo(vec);
			fObject->SetField(vec.GetThreeVector());
		}
		else
		{
			return false;
		}
		return true;
	}

	template< >
	inline bool KElectrostaticConstantFieldBuilder::AddElement(KContainer* /*aContainer*/)
	{
		return false;
	}

} //katrin



#endif /* KELECTROSTATICCONSTANTFIELDBUILDER_HH_ */
