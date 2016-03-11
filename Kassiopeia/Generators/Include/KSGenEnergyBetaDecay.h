//
// Created by trost on 29.05.15.
//

#ifndef KASPER_KSGENENERGYBETADECAY_H
#define KASPER_KSGENENERGYBETADECAY_H

#include "KSGenCreator.h"
#include "KField.h"

namespace Kassiopeia
{
    class KSGenEnergyBetaDecay:
            public KSComponentTemplate< KSGenEnergyBetaDecay, KSGenCreator >
    {
    public:
        KSGenEnergyBetaDecay();
        KSGenEnergyBetaDecay( const KSGenEnergyBetaDecay& aCopy );
        KSGenEnergyBetaDecay* Clone() const;
        virtual ~KSGenEnergyBetaDecay();

        //******
        //action
        //******

    public:
        void Dice( KSParticleQueue* aPrimaries );

        //*************
        //configuration
        //*************

    public:
        double Fermi(double E,double mnu, double E0, double Z);
        double GetFermiMax(double E0, double mnu, double Z);
        double GenBetaEnergy(double E0, double mnu, double Fermimax, double Z);

    private:
        ;K_SET_GET( int, ZDaughter );
        ;K_SET_GET( int, nmax );
        ;K_SET_GET( double, FermiMax );
        ;K_SET_GET( double, Endpoint );
        ;K_SET_GET( double, mnu );
        ;K_SET_GET( double, MinEnergy );
        ;K_SET_GET( double, MaxEnergy );

        //**********
        //initialize
        //**********

    public:
        void InitializeComponent();
        void DeinitializeComponent();
    };

}

#endif //KASPER_KSGENENERGYBETADECAY_H
