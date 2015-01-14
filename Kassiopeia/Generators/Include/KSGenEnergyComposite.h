#ifndef Kassiopeia_KSGenEnergyComposite_h_
#define Kassiopeia_KSGenEnergyComposite_h_

#include "KSGenCreator.h"
#include "KSGenValue.h"

namespace Kassiopeia
{

    class KSGenEnergyComposite :
        public KSComponentTemplate< KSGenEnergyComposite, KSGenCreator >
    {
        public:
            KSGenEnergyComposite();
            KSGenEnergyComposite( const KSGenEnergyComposite& aCopy );
            KSGenEnergyComposite* Clone() const;
            virtual ~KSGenEnergyComposite();

        public:
            virtual void Dice( KSParticleQueue* aPrimaries );

        public:
            void SetEnergyValue( KSGenValue* anEnergyValue );
            void ClearEnergyValue( KSGenValue* anEnergyValue );

        private:
            KSGenValue* fEnergyValue;

        protected:
            void InitializeComponent();
            void DeinitializeComponent();
    };
}

#endif
