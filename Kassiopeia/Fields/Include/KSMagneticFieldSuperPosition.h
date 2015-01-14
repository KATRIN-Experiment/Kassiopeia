#ifndef Kassiopeia_KSMagneticFieldSuperPosition_h_
#define Kassiopeia_KSMagneticFieldSuperPosition_h_

#include "KField.h"

#include "KSMagneticField.h"
#include "KThreeVector.h"
using katrin::KThreeVector;
#include "KThreeMatrix.h"
using katrin::KThreeMatrix;
#include <vector>
using std::vector;
#include <map>
using std::map;

namespace Kassiopeia
{

    class KSMagneticFieldSuperPosition :
        public KSMagneticField
    {
        public:
            KSMagneticFieldSuperPosition();
            virtual ~KSMagneticFieldSuperPosition();

            virtual bool GetField( KThreeVector& aTarget, const KThreeVector& aSamplePoint, const double& aSampleTime );
            virtual bool GetGradient( KThreeMatrix& aTarget, const KThreeVector& aSamplePoint, const Double_t& aSampleTime );

            bool GetCachedField( KThreeVector& aTarget, const KThreeVector& aSamplePoint, const Double_t& aSampleTime );
            bool GetCachedGradient( KThreeMatrix& aTarget, const KThreeVector& aSamplePoint, const Double_t& aSampleTime );

            void SetEnhancements( vector< double > aEnhancementVector );
            vector< double > GetEnhancements();

            void SetUseCaching( bool aBool );

            void AddMagneticField( KSMagneticField* aField, double aEnhancement = 1.0 );

        private:
            vector< KSMagneticField* > fMagneticFields;
            vector< double > fEnhancements;

            bool fUseCaching;
            map < KThreeVector, vector<KThreeVector> > fFieldCache;
            map < KThreeVector, vector<KThreeMatrix> > fGradientCache;
    };

}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////                                                   /////
/////  BBBB   U   U  IIIII  L      DDDD   EEEEE  RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB   U   U    I    L      D   D  EE     RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB    UUU   IIIII  LLLLL  DDDD   EEEEE  R   R  /////
/////                                                   /////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

#include "KComplexElement.hh"
#include "KSFieldToolbox.h"

using namespace Kassiopeia;
namespace katrin
{
    class KSMagneticFieldSuperPositionData
    {
        public:
            string fField;
            double fEnhancement;
    };


    typedef KComplexElement< KSMagneticFieldSuperPositionData > KSMagneticFieldSuperPositionDataBuilder;

    template< >
    inline bool KSMagneticFieldSuperPositionDataBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "field" )
        {
            aContainer->CopyTo( fObject->fField );
            return true;
        }
        if( aContainer->GetName() == "enhancement" )
        {
            aContainer->CopyTo( fObject->fEnhancement );
            return true;
        }
        return false;
    }



    typedef KComplexElement< KSMagneticFieldSuperPosition > KSMagneticFieldSuperPositionBuilder;

    template< >
    inline bool KSMagneticFieldSuperPositionBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "use_caching" )
        {
            aContainer->CopyTo( fObject, &KSMagneticFieldSuperPosition::SetUseCaching );
            return true;
        }
        return false;
    }

    template< >
    inline bool KSMagneticFieldSuperPositionBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->Is< KSMagneticFieldSuperPositionData >() == true )
        {
            KSMagneticFieldSuperPositionData* tMagneticFieldSuperPositionData = aContainer->AsPointer< KSMagneticFieldSuperPositionData >();
            KSMagneticField* tMagneticField = KSFieldToolbox::GetInstance()->GetObject< KSMagneticField >( tMagneticFieldSuperPositionData->fField );
            fObject->AddMagneticField( tMagneticField, tMagneticFieldSuperPositionData->fEnhancement );
            return true;
        }
        return false;
    }



}

#endif //KSMAGNETICFIELDSUPERPOSITION_H_
