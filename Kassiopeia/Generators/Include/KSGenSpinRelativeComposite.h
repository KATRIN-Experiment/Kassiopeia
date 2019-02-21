#ifndef Kassiopeia_KSGenSpinRelativeComposite_h_
#define Kassiopeia_KSGenSpinRelativeComposite_h_

#include "KSGenCreator.h"
#include "KSGenValue.h"

namespace Kassiopeia
{

    class KSGenSpinRelativeComposite :
        public KSComponentTemplate< KSGenSpinRelativeComposite, KSGenCreator >
    {
        public:
            KSGenSpinRelativeComposite();
            KSGenSpinRelativeComposite( const KSGenSpinRelativeComposite& aCopy );
            KSGenSpinRelativeComposite* Clone() const;
            virtual ~KSGenSpinRelativeComposite();

        public:
            virtual void Dice( KSParticleQueue* aParticleList );

        public:
            void SetThetaValue( KSGenValue* anThetaValue );
            void ClearThetaValue( KSGenValue* anThetaValue );

            void SetPhiValue( KSGenValue* aPhiValue );
            void ClearPhiValue( KSGenValue* aPhiValue );

        private:
            KSGenValue* fThetaValue;
            KSGenValue* fPhiValue;

        protected:
            void InitializeComponent();
            void DeinitializeComponent();
    };

}

#endif
