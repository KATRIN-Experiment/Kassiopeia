#include "KSGenPositionFrustrumComposite.h"
#include "KSGeneratorsMessage.h"

using namespace std;

namespace Kassiopeia
{

    KSGenPositionFrustrumComposite::KSGenPositionFrustrumComposite()
    {
        fCoordinateMap[eRadius] = 0;
        fCoordinateMap[ePhi] = 1;
        fCoordinateMap[eZ] = 2;
    }
    KSGenPositionFrustrumComposite::KSGenPositionFrustrumComposite( const KSGenPositionFrustrumComposite& aCopy ) :
            KSComponent(),
            fCoordinateMap( aCopy.fCoordinateMap ),
            fValues( aCopy.fValues )
    {
    }
    KSGenPositionFrustrumComposite* KSGenPositionFrustrumComposite::Clone() const
    {
        return new KSGenPositionFrustrumComposite( *this );
    }
    KSGenPositionFrustrumComposite::~KSGenPositionFrustrumComposite()
    {
    }

    void KSGenPositionFrustrumComposite::Dice( KSParticleQueue* aPrimaries )
    {
        bool tHasRValue = false;
        bool tHasPhiValue = false;
        bool tHasZValue = false;

        for( vector<pair<CoordinateType,KSGenValue*> >::iterator tIt = fValues.begin(); tIt != fValues.end(); tIt++)
        {
            tHasRValue = tHasRValue | ( (*tIt).first == eRadius );
            tHasPhiValue = tHasPhiValue | ( (*tIt).first == ePhi );
            tHasZValue = tHasZValue | ( (*tIt).first == eZ );
        }

        if ( !tHasRValue | !tHasPhiValue | !tHasZValue )
            genmsg( eError ) << "r, phi or z value undefined in composite position creator <"
                             << this->GetName() << ">" << eom;

        KThreeVector tPosition;
        KThreeVector tFrustrumPosition;

        KSParticle* tParticle;
        KSParticleIt tParticleIt;
        KSParticleQueue tParticles;

        vector< double > tFirstValues;
        vector< double >::iterator tFirstValueIt;
        vector< double > tSecondValues;
        vector< double >::iterator tSecondValueIt;
        vector< double > tThirdValues;
        vector< double >::iterator tThirdValueIt;

        fValues.at(0).second->DiceValue( tFirstValues );
        fValues.at(1).second->DiceValue( tSecondValues );
        fValues.at(2).second->DiceValue( tThirdValues );

        for( tFirstValueIt = tFirstValues.begin(); tFirstValueIt != tFirstValues.end(); tFirstValueIt++ )
        {
            tFrustrumPosition[fCoordinateMap.at(fValues.at(0).first)] = (*tFirstValueIt);

            for( tSecondValueIt = tSecondValues.begin(); tSecondValueIt != tSecondValues.end(); tSecondValueIt++ )
            {
                tFrustrumPosition[fCoordinateMap.at(fValues.at(1).first)] = (*tSecondValueIt);

                for( tThirdValueIt = tThirdValues.begin(); tThirdValueIt != tThirdValues.end(); tThirdValueIt++ )
                {
                    tFrustrumPosition[fCoordinateMap.at(fValues.at(2).first)] = (*tThirdValueIt);

                    for( tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++ )
                    {
                        double tPhiValue = (KConst::Pi() / 180.) * tFrustrumPosition[1];
                        double tZValue = tFrustrumPosition[2];
                        double tRValue = tFrustrumPosition[0] * ( r1 + ( tZValue - z1 ) / ( z2 - z1 ) * (r2 - r1) );

                        tParticle = new KSParticle( **tParticleIt );

                        tPosition = KThreeVector( 0., 0., 0. );
                        tPosition += tRValue * cos( tPhiValue ) * KThreeVector( 1., 0., 0. );
                        tPosition += tRValue * sin( tPhiValue ) * KThreeVector( 0., 1., 0. );
                        tPosition += tZValue * KThreeVector( 0., 0., 1. );

                        tParticle->SetPosition( tPosition );
                        tParticles.push_back( tParticle );

                    }
                }
            }
        }

        for( tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++ )
        {
            tParticle = *tParticleIt;
            delete tParticle;
        }

        aPrimaries->assign( tParticles.begin(), tParticles.end() );

        return;
    }

    void KSGenPositionFrustrumComposite::SetRValue( KSGenValue* anRValue )
    {

        for( vector<pair<CoordinateType,KSGenValue*> >::iterator tIt = fValues.begin(); tIt != fValues.end(); tIt++)
        {
            if( (*tIt).first == eRadius )
            {
                genmsg( eError ) << "cannot set r value <" << anRValue->GetName()
                                 << "> to composite position frustrum creator <"
                                 << this->GetName() << ">" << eom;
                return;
            }
        }
        fValues.push_back(std::pair<CoordinateType,KSGenValue*>(eRadius,anRValue));
    }

    void KSGenPositionFrustrumComposite::ClearRValue( KSGenValue* anRValue )
    {
        for( vector<pair<CoordinateType,KSGenValue*> >::iterator tIt = fValues.begin(); tIt != fValues.end(); tIt++)
        {
            if( (*tIt).first == eRadius )
            {
                fValues.erase( tIt );
                return;
            }
        }

        genmsg( eError ) << "cannot clear r value <" << anRValue->GetName()
                         << "> from composite position frustrum creator <"
                         << this->GetName() << ">" << eom;
        return;
    }

    void KSGenPositionFrustrumComposite::SetPhiValue( KSGenValue* aPhiValue )
    {
        for( vector<pair<CoordinateType,KSGenValue*> >::iterator tIt = fValues.begin(); tIt != fValues.end(); tIt++)
        {
            if( (*tIt).first == ePhi )
            {
                genmsg( eError ) << "cannot set phi value <" << aPhiValue->GetName()
                                 << "> to composite position frustrum creator <"
                                 << this->GetName() << ">" << eom;
                return;
            }
        }
        fValues.push_back(std::pair<CoordinateType,KSGenValue*>(ePhi,aPhiValue));
        return;
    }
    void KSGenPositionFrustrumComposite::ClearPhiValue( KSGenValue* anPhiValue )
    {
        for( vector<pair<CoordinateType,KSGenValue*> >::iterator tIt = fValues.begin(); tIt != fValues.end(); tIt++)
        {
            if( (*tIt).first == ePhi )
            {
                fValues.erase( tIt );
                return;
            }
        }

        genmsg( eError ) << "cannot clear phi value <" << anPhiValue->GetName()
                         << "> from composite position frustrum creator <"
                         << this->GetName() << ">" << eom;
        return;
    }

    void KSGenPositionFrustrumComposite::SetZValue( KSGenValue* anZValue )
    {
        for( vector<pair<CoordinateType,KSGenValue*> >::iterator tIt = fValues.begin(); tIt != fValues.end(); tIt++)
        {
            if( (*tIt).first == eZ )
            {
                genmsg( eError ) << "cannot set z value <" << anZValue->GetName()
                                 << "> to composite position frustrum creator <"
                                 << this->GetName() << ">" << eom;
                return;
            }
        }
        fValues.push_back(std::pair<CoordinateType,KSGenValue*>(eZ,anZValue));

    }
    void KSGenPositionFrustrumComposite::ClearZValue( KSGenValue* anZValue )
    {
        for( vector<pair<CoordinateType,KSGenValue*> >::iterator tIt = fValues.begin(); tIt != fValues.end(); tIt++)
        {
            if( (*tIt).first == eZ )
            {
                fValues.erase( tIt );
                return;
            }
        }

        genmsg( eError ) << "cannot clear z value <" << anZValue->GetName()
                         << "> from composite position frustrum creator <"
                         << this->GetName() << ">" << eom;
        return;
    }

    void KSGenPositionFrustrumComposite::SetR1Value( KSGenValue* anRValue )
    {
        vector<double> temp;
        anRValue->DiceValue( temp );
        r1 = temp[0];
        return;
    }

    void KSGenPositionFrustrumComposite::SetR2Value( KSGenValue* anRValue )
    {
        vector<double> temp;
        anRValue->DiceValue( temp );
        r2 = temp[0];
        return;
    }

    void KSGenPositionFrustrumComposite::SetZ1Value( KSGenValue* aZValue )
    {
        vector<double> temp;
        aZValue->DiceValue( temp );
        z1 = temp[0];
        return;
    }

    void KSGenPositionFrustrumComposite::SetZ2Value( KSGenValue* aZValue )
    {
        vector<double> temp;
        aZValue->DiceValue( temp );
        z2 = temp[0];
        return;
    }

    // void KSGenPositionFrustrumComposite::SetR1Value( double anR1Value )
    // {
    //     std::cout << "R1" << "\t" << anR1Value << "\n";
    //     r1 = anR1Value;
    // }
    // void KSGenPositionFrustrumComposite::SetZ1Value( double aZ1Value )
    // {
    //     std::cout << "Z1" << "\t" << aZ1Value<< "\n";
    //     z1 = aZ1Value;
    // }
    // void KSGenPositionFrustrumComposite::SetR2Value( double anR2Value )
    // {
    //     std::cout << "R2" << "\t" << anR2Value<< "\n";
    //     r2 = anR2Value;
    // }
    // void KSGenPositionFrustrumComposite::SetZ2Value( double aZ2Value )
    // {
    //     std::cout << "Z2" << "\t" << aZ2Value<< "\n";
    //     z2 = aZ2Value;
    // }

    void KSGenPositionFrustrumComposite::InitializeComponent()
    {
        for( vector<pair<CoordinateType,KSGenValue*> >::iterator tIt = fValues.begin(); tIt != fValues.end(); tIt++)
        {
            (*tIt).second->Initialize();
        }
        return;
    }
    void KSGenPositionFrustrumComposite::DeinitializeComponent()
    {
        for( vector<pair<CoordinateType,KSGenValue*> >::iterator tIt = fValues.begin(); tIt != fValues.end(); tIt++)
        {
            (*tIt).second->Deinitialize();
        }
        return;
    }

    STATICINT sKSGenPositionFrustrumCompositeDict =
        // KSDictionary< KSGenPositionFrustrumComposite >::AddCommand( &KSGenPositionFrustrumComposite::SetR1Value,
        //                                                                &KSGenPositionFrustrumComposite::SetR2Value,
        //                                                                "set_r1", "set_r2" ) +
        // KSDictionary< KSGenPositionFrustrumComposite >::AddCommand( &KSGenPositionFrustrumComposite::SetZ1Value,
        //                                                               &KSGenPositionFrustrumComposite::SetZ2Value,
        //                                                               "set_z1", "set_z2" ) +
        KSDictionary< KSGenPositionFrustrumComposite >::AddCommand( &KSGenPositionFrustrumComposite::SetRValue,
                                                                       &KSGenPositionFrustrumComposite::ClearRValue,
                                                                       "set_r", "clear_r" ) +
        KSDictionary< KSGenPositionFrustrumComposite >::AddCommand( &KSGenPositionFrustrumComposite::SetPhiValue,
                                                                       &KSGenPositionFrustrumComposite::ClearPhiValue,
                                                                       "set_phi", "clear_phi" ) +
        KSDictionary< KSGenPositionFrustrumComposite >::AddCommand( &KSGenPositionFrustrumComposite::SetZValue,
                                                                       &KSGenPositionFrustrumComposite::ClearZValue,
                                                                       "set_z", "clear_z" );

}
