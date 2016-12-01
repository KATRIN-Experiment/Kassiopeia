#ifndef Kassiopeia_KSGenPositionFrustrumComposite_h_
#define Kassiopeia_KSGenPositionFrustrumComposite_h_

#include "KSGenCreator.h"
#include "KSGenValue.h"

namespace Kassiopeia
{



    class KSGenPositionFrustrumComposite :
        public KSComponentTemplate< KSGenPositionFrustrumComposite, KSGenCreator >
    {
        public:
            KSGenPositionFrustrumComposite();
            KSGenPositionFrustrumComposite( const KSGenPositionFrustrumComposite& aCopy );
            KSGenPositionFrustrumComposite* Clone() const;
            virtual ~KSGenPositionFrustrumComposite();

        public:
            virtual void Dice( KSParticleQueue* aPrimaryList );

        public:

            void SetRValue( KSGenValue* anRValue );
            void ClearRValue( KSGenValue* anRValue );

            void SetPhiValue( KSGenValue* aPhiValue );
            void ClearPhiValue( KSGenValue* aPhiValue );

            void SetZValue( KSGenValue* anZValue );
            void ClearZValue( KSGenValue* anZValue );

            void SetR1Value( KSGenValue* anRValue );
            void SetR2Value( KSGenValue* anRValue );

            void SetZ1Value( KSGenValue* anZValue );
            void SetZ2Value( KSGenValue* anZValue );

            // void SetR1Value( double r1 );
            // void SetZ1Value( double z1 );
            //
            // void SetR2Value( double r2 );
            // void SetZ2Value( double z2 );

        private:

            double r1;
            double z1;

            double r2;
            double z2;

            typedef enum
            {
                eRadius, ePhi, eZ
            } CoordinateType;

            std::map<CoordinateType, int> fCoordinateMap;
            std::vector<std::pair<CoordinateType,KSGenValue*> > fValues;


        protected:
            void InitializeComponent();
            void DeinitializeComponent();
    };

}

#endif
