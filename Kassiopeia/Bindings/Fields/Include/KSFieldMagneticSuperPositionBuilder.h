#ifndef Kassiopeia_KSFieldMagneticSuperPositionBuilder_h_
#define Kassiopeia_KSFieldMagneticSuperPositionBuilder_h_

#include "KComplexElement.hh"
#include "KSFieldMagneticSuperPosition.h"
#include "KSToolbox.h"

using namespace Kassiopeia;
namespace katrin
{
	class KSFieldMagneticSuperPositionData
	{
		public:
			string fName;
			double fEnhancement;
	};


	typedef KComplexElement< KSFieldMagneticSuperPositionData > KSFieldMagneticSuperPositionDataBuilder;

	template< >
	inline bool KSFieldMagneticSuperPositionDataBuilder::AddAttribute( KContainer* aContainer )
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


    typedef KComplexElement< KSFieldMagneticSuperPosition > KSFieldMagneticSuperPositionBuilder;

    template< >
    inline bool KSFieldMagneticSuperPositionBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KSMagneticField::SetName );
            return true;
        }
        if( aContainer->GetName() == "use_caching" )
        {
            aContainer->CopyTo( fObject, &KSFieldMagneticSuperPosition::SetUseCaching );
            return true;
        }
        return false;
    }

    template< >
    inline bool KSFieldMagneticSuperPositionBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->Is< KSFieldMagneticSuperPositionData >() == true )
        {
        	KSFieldMagneticSuperPositionData* tFieldMagneticSuperPositionData = aContainer->AsPointer< KSFieldMagneticSuperPositionData >();
        	KSMagneticField* tMagneticField = KSToolbox::GetInstance()->GetObjectAs< KSMagneticField >( tFieldMagneticSuperPositionData->fName );
            fObject->AddMagneticField( tMagneticField, tFieldMagneticSuperPositionData->fEnhancement );
            return true;
        }
        return false;
    }




}

#endif
