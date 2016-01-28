#ifndef Kassiopeia_KSFieldMagneticSuperPosition_h_
#define Kassiopeia_KSFieldMagneticSuperPosition_h_

#include "KSMagneticField.h"

#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

#include "KThreeMatrix.hh"
using KGeoBag::KThreeMatrix;

#include "KField.h"

#include <vector>
using std::vector;

#include <map>
using std::map;

namespace Kassiopeia
{

    class KSFieldMagneticSuperPosition :
		public KSComponentTemplate< KSFieldMagneticSuperPosition, KSMagneticField >
    {
        public:
    		KSFieldMagneticSuperPosition();
    		KSFieldMagneticSuperPosition( const KSFieldMagneticSuperPosition& aCopy );
    		KSFieldMagneticSuperPosition* Clone() const;
            virtual ~KSFieldMagneticSuperPosition();

            void CalculateField( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField );
            void CalculateGradient( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeMatrix& aGradient );

            void CalculateCachedField( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField );
            void CalculateCachedGradient( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeMatrix& aGradient );

            void SetEnhancements( vector< double > aEnhancementVector );
            vector< double > GetEnhancements();

            void AddMagneticField( KSMagneticField* aField, double aEnhancement = 1.0 );

        private:
            void InitializeComponent();
            void DeinitializeComponent();

        private:
            vector< KSMagneticField* > fMagneticFields;
            vector< double > fEnhancements;

            ;K_SET( bool, UseCaching );
            map < KThreeVector, vector<KThreeVector> > fFieldCache;
            map < KThreeVector, vector<KThreeMatrix> > fGradientCache;
    };

}

#endif //Kassiopeia_KSFieldMagneticSuperPosition_h_
