#include "KSFieldMagneticSuperPosition.h"
#include "KSFieldsMessage.h"

namespace Kassiopeia
{
	KSFieldMagneticSuperPosition::KSFieldMagneticSuperPosition() :
        fMagneticFields(),
        fEnhancements(),
        fUseCaching( false ),
        fFieldCache(),
        fGradientCache()
    {
    }

	KSFieldMagneticSuperPosition::KSFieldMagneticSuperPosition( const KSFieldMagneticSuperPosition& aCopy ) :
		KSComponent(aCopy),
		fMagneticFields( aCopy.fMagneticFields ),
		fEnhancements( aCopy.fEnhancements ),
		fUseCaching( aCopy.fUseCaching ),
		fFieldCache( aCopy.fFieldCache ),
		fGradientCache( aCopy.fGradientCache )
    {
    }
	KSFieldMagneticSuperPosition* KSFieldMagneticSuperPosition::Clone() const
    {
        return new KSFieldMagneticSuperPosition( *this );
    }

	KSFieldMagneticSuperPosition::~KSFieldMagneticSuperPosition()
    {
    }

    void KSFieldMagneticSuperPosition::CalculateField( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField  )
    {
    	aField = KThreeVector::sZero;
        if ( fUseCaching )
        {
            return ( CalculateCachedField( aSamplePoint, aSampleTime, aField ) );
        }

        for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++)
        {
            KThreeVector tCurrentField(0.0,0.0,0.0);
            fMagneticFields.at( tIndex )->CalculateField( aSamplePoint, aSampleTime, tCurrentField );
            aField += tCurrentField * fEnhancements.at( tIndex );
            fieldmsg_debug( "B of <"<< fMagneticFields.at( tIndex )->GetName() << "> is " << tCurrentField * fEnhancements.at( tIndex )<<eom );
        }
        fieldmsg_debug( " Returning B of "<<aField<<eom );
    }

    void KSFieldMagneticSuperPosition::CalculateCachedField( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField )
    {
    	aField = KThreeVector::sZero;
        //looking in cache for aSamplePoint
        std::map< KThreeVector, vector< KThreeVector> >::iterator tCacheIt;
        for ( tCacheIt=fFieldCache.begin(); tCacheIt != fFieldCache.end(); tCacheIt++)
        {
            if(tCacheIt->first == aSamplePoint)
            {
                for (size_t tIndex = 0; tIndex < tCacheIt->second.size(); tIndex++)
                {
                    aField += tCacheIt->second.at( tIndex ) * fEnhancements.at( tIndex );
                }
                return;
            }
        }

        //Calculating Fields without Enhancement for aSamplePoint and insert it into the cache
        vector<KThreeVector> tFields;
        KThreeVector tCurrentField(0.0,0.0,0.0);
        for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++)
        {
            KThreeVector tCurrentField(0.0,0.0,0.0);
            fMagneticFields.at( tIndex )->CalculateField( aSamplePoint, aSampleTime, tCurrentField );
            aField += tCurrentField * fEnhancements.at( tIndex );
            tFields.push_back( tCurrentField );
        }
        fFieldCache.insert( pair<KThreeVector, vector< KThreeVector> > ( aSamplePoint, tFields ) );
        return;
    }

    void KSFieldMagneticSuperPosition::CalculateGradient( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeMatrix& aGradient  )
    {
    	aGradient = KThreeMatrix::sZero;
        if ( fUseCaching )
        {
            return ( CalculateCachedGradient( aSamplePoint, aSampleTime, aGradient ) );
        }

        for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++)
        {
            KThreeMatrix tCurrentGradient;
            fMagneticFields.at( tIndex )->CalculateGradient( aSamplePoint, aSampleTime, tCurrentGradient );
            aGradient += tCurrentGradient * fEnhancements.at( tIndex );
        }
    }

    void KSFieldMagneticSuperPosition::CalculateCachedGradient( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeMatrix& aGradient )
    {
    	aGradient = KThreeMatrix::sZero;
        //looking in cache for aSamplePoint
        std::map< KThreeVector, vector< KThreeMatrix> >::iterator tCacheIt;
        for ( tCacheIt=fGradientCache.begin(); tCacheIt != fGradientCache.end(); tCacheIt++)
        {
            if(tCacheIt->first == aSamplePoint)
            {
                for (size_t tIndex = 0; tIndex < tCacheIt->second.size(); tIndex++)
                {
                	aGradient += tCacheIt->second.at( tIndex ) * fEnhancements.at( tIndex );
                }
                return;
            }
        }

        //Calculating Fields without Enhancement for aSamplePoint and insert it into the cache
        vector<KThreeMatrix> tGradients;
        KThreeMatrix tCurrentGradient;
        for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++)
        {
            KThreeMatrix tCurrentGradient;
            fMagneticFields.at( tIndex )->CalculateGradient( aSamplePoint, aSampleTime, tCurrentGradient );
            aGradient += tCurrentGradient * fEnhancements.at( tIndex );
            tGradients.push_back( tCurrentGradient );
        }
        fGradientCache.insert( pair<KThreeVector, vector< KThreeMatrix> > ( aSamplePoint, tGradients ) );
        return;
    }


    void KSFieldMagneticSuperPosition::SetEnhancements( vector< double > aEnhancementVector )
    {
        if ( aEnhancementVector.size() != fEnhancements.size() )
        {
            fieldmsg( eError ) <<"EnhancementVector has not the same size <"<<aEnhancementVector.size()<<"> as fEnhancements <"<<fEnhancements.size()<<">"<<eom;
        }
        fEnhancements = aEnhancementVector;
        return;
    }

    vector< double> KSFieldMagneticSuperPosition::GetEnhancements()
    {
        return fEnhancements;
    }

    void KSFieldMagneticSuperPosition::AddMagneticField(KSMagneticField* aField, double aEnhancement)
    {
        fieldmsg_debug( "adding field with key: "<<aField->GetName() << eom );
        fMagneticFields.push_back( aField );
        fEnhancements.push_back( aEnhancement );
        return;
    }

    void KSFieldMagneticSuperPosition::InitializeComponent()
    {
        for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++)
        {
            fMagneticFields.at( tIndex )->Initialize();
        }
    }

    void KSFieldMagneticSuperPosition::DeinitializeComponent()
    {
        for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++)
        {
            fMagneticFields.at( tIndex )->Deinitialize();
        }
    }

}
