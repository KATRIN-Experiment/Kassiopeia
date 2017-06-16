#include "KSGenPositionSphericalComposite.h"
#include "KSGeneratorsMessage.h"

using namespace std;

namespace Kassiopeia
{

    KSGenPositionSphericalComposite::KSGenPositionSphericalComposite() :
            fOrigin( KThreeVector::sZero ),
            fXAxis( KThreeVector::sXUnit ),
            fYAxis( KThreeVector::sYUnit ),
            fZAxis( KThreeVector::sZUnit )
    {
        fCoordinateMap[eRadius] = 0;
        fCoordinateMap[eTheta] = 1;
        fCoordinateMap[ePhi] = 2;
    }
    KSGenPositionSphericalComposite::KSGenPositionSphericalComposite( const KSGenPositionSphericalComposite& aCopy ) :
            KSComponent(),
            fOrigin( aCopy.fOrigin ),
            fXAxis( aCopy.fXAxis ),
            fYAxis( aCopy.fYAxis ),
            fZAxis( aCopy.fZAxis ),
            fCoordinateMap( aCopy.fCoordinateMap ),
            fValues( aCopy.fValues )
    {
    }
    KSGenPositionSphericalComposite* KSGenPositionSphericalComposite::Clone() const
    {
        return new KSGenPositionSphericalComposite( *this );
    }
    KSGenPositionSphericalComposite::~KSGenPositionSphericalComposite()
    {
    }

    void KSGenPositionSphericalComposite::Dice( KSParticleQueue* aPrimaries )
    {
        bool tHasRValue = false;
        bool tHasThetaValue = false;
        bool tHasPhiValue = false;

        for( vector<pair<CoordinateType,KSGenValue*> >::iterator tIt = fValues.begin(); tIt != fValues.end(); tIt++)
        {
            tHasRValue = tHasRValue | ( (*tIt).first == eRadius );
            tHasThetaValue = tHasThetaValue | ( (*tIt).first == eTheta );
            tHasPhiValue = tHasPhiValue | ( (*tIt).first == ePhi );
        }

        if ( !tHasRValue | !tHasThetaValue | !tHasPhiValue )
            genmsg( eError ) << "r, phi or z value undefined in composite position creator <"
                             << this->GetName() << ">" << eom;

        KThreeVector tPosition;
        KThreeVector tSphericalPosition;

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
            tSphericalPosition[fCoordinateMap.at(fValues.at(0).first)] = (*tFirstValueIt);

            for( tSecondValueIt = tSecondValues.begin(); tSecondValueIt != tSecondValues.end(); tSecondValueIt++ )
            {
                tSphericalPosition[fCoordinateMap.at(fValues.at(1).first)] = (*tSecondValueIt);

                for( tThirdValueIt = tThirdValues.begin(); tThirdValueIt != tThirdValues.end(); tThirdValueIt++ )
                {
                    tSphericalPosition[fCoordinateMap.at(fValues.at(2).first)] = (*tThirdValueIt);

                    for( tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++ )
                    {
                        double tRValue = tSphericalPosition[0];
                        double tThetaValue = tSphericalPosition[1];
                        double tPhiValue = tSphericalPosition[2];

                        tParticle = new KSParticle( **tParticleIt );
                        tPosition = fOrigin;
                        tPosition += tRValue * sin( tThetaValue ) * cos( tPhiValue ) * fXAxis;
                        tPosition += tRValue * sin( tThetaValue ) * sin( tPhiValue ) * fYAxis;
                        tPosition += tRValue * cos( tThetaValue ) * fZAxis;
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

    void KSGenPositionSphericalComposite::SetOrigin( const KThreeVector& anOrigin )
    {
        fOrigin = anOrigin;
        return;
    }
    void KSGenPositionSphericalComposite::SetXAxis( const KThreeVector& anXAxis )
    {
        fXAxis = anXAxis;
        return;
    }
    void KSGenPositionSphericalComposite::SetYAxis( const KThreeVector& anYAxis )
    {
        fYAxis = anYAxis;
        return;
    }
    void KSGenPositionSphericalComposite::SetZAxis( const KThreeVector& anZAxis )
    {
        fZAxis = anZAxis;
        return;
    }

    void KSGenPositionSphericalComposite::SetRValue( KSGenValue* anRValue )
    {
        for( vector<pair<CoordinateType,KSGenValue*> >::iterator tIt = fValues.begin(); tIt != fValues.end(); tIt++)
        {
            if( (*tIt).first == eRadius )
            {
                genmsg( eError ) << "cannot set r value <" << anRValue->GetName()
                                 << "> to composite position spherical creator <"
                                 << this->GetName() << ">" << eom;
                return;
            }
        }
        fValues.push_back(std::pair<CoordinateType,KSGenValue*>(eRadius,anRValue));
    }
    void KSGenPositionSphericalComposite::ClearRValue( KSGenValue* anRValue )
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
                         << "> from composite position spherical creator <"
                         << this->GetName() << ">" << eom;
        return;
    }

    void KSGenPositionSphericalComposite::SetThetaValue( KSGenValue* anThetaValue )
    {
        for( vector<pair<CoordinateType,KSGenValue*> >::iterator tIt = fValues.begin(); tIt != fValues.end(); tIt++)
        {
            if( (*tIt).first == eTheta )
            {
                genmsg( eError ) << "cannot set theta value <" << anThetaValue->GetName()
                                 << "> to composite position spherical creator <"
                                 << this->GetName() << ">" << eom;
                return;
            }
        }
        fValues.push_back(std::pair<CoordinateType,KSGenValue*>(eTheta,anThetaValue));
        return;
    }
    void KSGenPositionSphericalComposite::ClearThetaValue( KSGenValue* anThetaValue )
    {
        for( vector<pair<CoordinateType,KSGenValue*> >::iterator tIt = fValues.begin(); tIt != fValues.end(); tIt++)
        {
            if( (*tIt).first == eTheta )
            {
                fValues.erase( tIt );
                return;
            }
        }

        genmsg( eError ) << "cannot clear theta value <" << anThetaValue->GetName()
                         << "> from composite position spherical creator <"
                         << this->GetName() << ">" << eom;
        return;
    }

    void KSGenPositionSphericalComposite::SetPhiValue( KSGenValue* aPhiValue )
    {
        for( vector<pair<CoordinateType,KSGenValue*> >::iterator tIt = fValues.begin(); tIt != fValues.end(); tIt++)
        {
            if( (*tIt).first == ePhi )
            {
                genmsg( eError ) << "cannot set phi value <" << aPhiValue->GetName()
                                 << "> to composite position spherical creator <"
                                 << this->GetName() << ">" << eom;
                return;
            }
        }
        fValues.push_back(std::pair<CoordinateType,KSGenValue*>(ePhi,aPhiValue));
        return;
    }
    void KSGenPositionSphericalComposite::ClearPhiValue( KSGenValue* anPhiValue )
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
                         << "> from composite position spherical creator <"
                         << this->GetName() << ">" << eom;
        return;
    }

    void KSGenPositionSphericalComposite::InitializeComponent()
    {
        for( vector<pair<CoordinateType,KSGenValue*> >::iterator tIt = fValues.begin(); tIt != fValues.end(); tIt++)
        {
            (*tIt).second->Initialize();
        }
        return;
    }
    void KSGenPositionSphericalComposite::DeinitializeComponent()
    {
        for( vector<pair<CoordinateType,KSGenValue*> >::iterator tIt = fValues.begin(); tIt != fValues.end(); tIt++)
        {
            (*tIt).second->Deinitialize();
        }
        return;
    }

    STATICINT sKSGenPositionSphericalCompositeDict =
        KSDictionary< KSGenPositionSphericalComposite >::AddCommand( &KSGenPositionSphericalComposite::SetRValue,
                                                                     &KSGenPositionSphericalComposite::ClearRValue,
                                                                     "set_r", "clear_r" ) +
        KSDictionary< KSGenPositionSphericalComposite >::AddCommand( &KSGenPositionSphericalComposite::SetThetaValue,
                                                                     &KSGenPositionSphericalComposite::ClearThetaValue,
                                                                     "set_theta", "clear_theta" ) +
        KSDictionary< KSGenPositionSphericalComposite >::AddCommand( &KSGenPositionSphericalComposite::SetPhiValue,
                                                                     &KSGenPositionSphericalComposite::ClearPhiValue,
                                                                     "set_phi", "clear_phi" );

}
