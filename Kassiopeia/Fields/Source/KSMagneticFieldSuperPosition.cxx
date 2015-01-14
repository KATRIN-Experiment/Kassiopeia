#include "KSMagneticFieldSuperPosition.h"
#include "KSFieldMessage.h"

namespace Kassiopeia
{
    KSMagneticFieldSuperPosition::KSMagneticFieldSuperPosition() :
        fMagneticFields(),
        fEnhancements(),
        fUseCaching( false ),
        fFieldCache(),
        fGradientCache()
    {
    }

    KSMagneticFieldSuperPosition::~KSMagneticFieldSuperPosition()
    {
    }

    bool KSMagneticFieldSuperPosition::GetField( KThreeVector& aTarget, const KThreeVector& aSamplePoint, const Double_t& aSampleTime )
    {
        aTarget.SetComponents( 0., 0., 0. );
        if ( fUseCaching )
        {
            return ( GetCachedField( aTarget, aSamplePoint, aSampleTime ) );
        }

        for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++)
        {
            KThreeVector tCurrentField(0.0,0.0,0.0);
            fMagneticFields.at( tIndex )->GetField( tCurrentField, aSamplePoint, aSampleTime );
            aTarget += tCurrentField * fEnhancements.at( tIndex );
            emfmsg_debug( "B_z of <"<< fMagneticFields.at( tIndex )->GetName() << "> is <" << tCurrentField.Z() * fEnhancements.at( tIndex )<<">"<<eom );
        }
        emfmsg_debug( " Returning B_z of <"<<aTarget.Z()<<">"<<eom );
        return true;
    }

    bool KSMagneticFieldSuperPosition::GetCachedField( KThreeVector& aTarget, const KThreeVector& aSamplePoint, const Double_t& aSampleTime )
    {
        //looking in cache for aSamplePoint
        std::map< KThreeVector, vector< KThreeVector> >::iterator tCacheIt;
        for ( tCacheIt=fFieldCache.begin(); tCacheIt != fFieldCache.end(); tCacheIt++)
        {
            if(tCacheIt->first == aSamplePoint)
            {
                for (size_t tIndex = 0; tIndex < tCacheIt->second.size(); tIndex++)
                {
                    KThreeVector tMagneticField = tCacheIt->second.at( tIndex );
                    aTarget += tMagneticField * fEnhancements.at( tIndex );
                }
                return true;
            }
        }

        //Calculating Fields without Enhancement for aSamplePoint and insert it into the cache
        vector<KThreeVector> tFields;
        for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++)
        {
            KThreeVector tCurrentField(0.0,0.0,0.0);
            fMagneticFields.at( tIndex )->GetField( tCurrentField, aSamplePoint, aSampleTime );
            tFields.push_back( tCurrentField );
        }
        fFieldCache.insert( pair<KThreeVector, vector< KThreeVector> > ( aSamplePoint, tFields ) );

        //this should now return true, as there is now a cache for aSamplePoint
        return ( GetCachedField( aTarget, aSamplePoint, aSampleTime ) );

    }


    bool KSMagneticFieldSuperPosition::GetGradient(KThreeMatrix& aTarget, const KThreeVector& aSamplePoint, const Double_t& aSampleTime )
    {
        aTarget = 0.0;
        if ( fUseCaching )
        {
            return ( GetCachedGradient( aTarget, aSamplePoint, aSampleTime ) );
        }

        for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++)
        {
            KThreeMatrix tCurrentGradient;
            fMagneticFields.at( tIndex )->GetGradient( tCurrentGradient, aSamplePoint, aSampleTime );
            aTarget += tCurrentGradient * fEnhancements.at( tIndex );
        }
        return true;
    }

    bool KSMagneticFieldSuperPosition::GetCachedGradient( KThreeMatrix& aTarget, const KThreeVector& aSamplePoint, const Double_t& aSampleTime )
    {
        //looking in cache for aSamplePoint
        std::map< KThreeVector, vector< KThreeMatrix> >::iterator tCacheIt;
        for ( tCacheIt=fGradientCache.begin(); tCacheIt != fGradientCache.end(); tCacheIt++)
        {
            if(tCacheIt->first == aSamplePoint)
            {
                for (size_t tIndex = 0; tIndex < tCacheIt->second.size(); tIndex++)
                {
                    KThreeMatrix tMagneticField = tCacheIt->second.at( tIndex );
                    aTarget += tMagneticField * fEnhancements.at( tIndex );
                }
                return true;
            }
        }

        //Calculating Fields without Enhancement for aSamplePoint and insert it into the cache
        vector<KThreeMatrix> tGradients;
        for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++)
        {
            KThreeMatrix tCurrentGradient;
            fMagneticFields.at( tIndex )->GetGradient( tCurrentGradient, aSamplePoint, aSampleTime );
            tGradients.push_back( tCurrentGradient );
        }
        fGradientCache.insert( pair<KThreeVector, vector< KThreeMatrix> > ( aSamplePoint, tGradients ) );

        //this should now return true, as there is now a cache for aSamplePoint
        return ( GetCachedGradient( aTarget, aSamplePoint, aSampleTime ) );

    }


    void KSMagneticFieldSuperPosition::SetEnhancements( vector< double > aEnhancementVector )
    {
        if ( aEnhancementVector.size() != fEnhancements.size() )
        {
            emfmsg( eError ) <<"EnhancementVector has not the same size <"<<aEnhancementVector.size()<<"> as fEnhancements <"<<fEnhancements.size()<<">"<<eom;
        }
        fEnhancements = aEnhancementVector;
        return;
    }

    vector< double> KSMagneticFieldSuperPosition::GetEnhancements()
    {
        return fEnhancements;
    }

    void KSMagneticFieldSuperPosition::SetUseCaching( bool aBool )
    {
        fUseCaching = aBool;
    }

    void KSMagneticFieldSuperPosition::AddMagneticField(KSMagneticField* aField, double aEnhancement)
    {
        emfmsg_debug( "adding field with key: "<<aField->GetName() << eom );
        fMagneticFields.push_back( aField );
        fEnhancements.push_back( aEnhancement );
        return;
    }

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


namespace katrin
{

    static int sMagneticFieldSuperPositionDataStructure =
        KSMagneticFieldSuperPositionDataBuilder::Attribute< string >( "field" ) +
        KSMagneticFieldSuperPositionDataBuilder::Attribute< double >( "enhancement" );

    static int sMagneticFieldSuperPositionData = KSMagneticFieldSuperPositionBuilder::ComplexElement< KSMagneticFieldSuperPositionData >( "add" );



    static int sMagneticFieldSuperPositionStructure =
        KSMagneticFieldSuperPositionBuilder::Attribute< string >( "name" ) +
        KSMagneticFieldSuperPositionBuilder::Attribute< bool >( "use_caching" );

    static int sMagneticFieldSuperPosition = KSFieldToolboxBuilder::ComplexElement< KSMagneticFieldSuperPosition >( "magnetic_field_super_position" );


}

