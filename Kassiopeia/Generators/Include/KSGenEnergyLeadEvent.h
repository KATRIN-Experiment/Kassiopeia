#ifndef KSGENENERGYLEADEVENT_H
#define KSGENENERGYLEADEVENT_H

#include "KSGenCreator.h"

namespace Kassiopeia
{
    class KSGenRelaxation;
    class KSGenConversion;

    class KSGenEnergyLeadEvent :
        public KSComponentTemplate< KSGenEnergyLeadEvent, KSGenCreator >
    {
        public:
            KSGenEnergyLeadEvent();
            KSGenEnergyLeadEvent( const KSGenEnergyLeadEvent& aCopy );
            KSGenEnergyLeadEvent* Clone() const;
            virtual ~KSGenEnergyLeadEvent();

            //******
            //action
            //******

        public:
            void Dice( KSParticleQueue* aPrimaries );

            //*************
            //configuration
            //*************

        public:
            void SetForceConversion( bool aSetting );
            void SetDoConversion( bool aSetting );
            void SetDoAuger( bool aSetting );
            double Fermi(double E,double mnu, double E0, double Z);
            double GetFermiMax(double E0, double mnu, double Z);
            double GenBetaEnergy(double E0, double mnu, double Fermimax, double Z);

        private:
            bool fForceConversion;
            bool fDoConversion;
//            bool fDoShakeOff;
            bool fDoAuger;
            int fIsotope;
            int fZDaughter;
            double fFermiMax17;
            double fFermiMax63;
            int fnmax;

            //**********
            //initialize
            //**********

        public:
            void InitializeComponent();
            void DeinitializeComponent();

        private:
            KSGenRelaxation* fBismuthRelaxation;
            KSGenConversion* fBismuthConversion;
    };

}

#endif // KSGENENERGYLEADEVENT_H
