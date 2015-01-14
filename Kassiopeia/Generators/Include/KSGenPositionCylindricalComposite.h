#ifndef Kassiopeia_KSGenPositionCylindricalComposite_h_
#define Kassiopeia_KSGenPositionCylindricalComposite_h_

#include "KSGenCreator.h"
#include "KSGenValue.h"

namespace Kassiopeia
{



    class KSGenPositionCylindricalComposite :
        public KSComponentTemplate< KSGenPositionCylindricalComposite, KSGenCreator >
    {
        public:
            KSGenPositionCylindricalComposite();
            KSGenPositionCylindricalComposite( const KSGenPositionCylindricalComposite& aCopy );
            KSGenPositionCylindricalComposite* Clone() const;
            virtual ~KSGenPositionCylindricalComposite();

        public:
            virtual void Dice( KSParticleQueue* aPrimaryList );

        public:
            void SetOrigin( const KThreeVector& anOrigin );
            void SetXAxis( const KThreeVector& anXAxis );
            void SetYAxis( const KThreeVector& anYAxis );
            void SetZAxis( const KThreeVector& anZAxis );

            void SetRValue( KSGenValue* anRValue );
            void ClearRValue( KSGenValue* anRValue );

            void SetPhiValue( KSGenValue* aPhiValue );
            void ClearPhiValue( KSGenValue* aPhiValue );

            void SetZValue( KSGenValue* anZValue );
            void ClearZValue( KSGenValue* anZValue );

        private:
            KThreeVector fOrigin;
            KThreeVector fXAxis;
            KThreeVector fYAxis;
            KThreeVector fZAxis;

            typedef enum
            {
                eRadius, ePhi, eZ
            } CoordinateType;

            std::map<CoordinateType, int> fCoordinateMap;
            vector<pair<CoordinateType,KSGenValue*> > fValues;


        protected:
            void InitializeComponent();
            void DeinitializeComponent();
    };

}

#endif
