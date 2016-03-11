#include "KSRootMagneticField.h"
#include "KSFieldsMessage.h"

namespace Kassiopeia
{

    KSRootMagneticField::KSRootMagneticField() :
            fCurrentField(),
            fCurrentGradient(),
            fMagneticFields( 128 )
    {
    }
    KSRootMagneticField::KSRootMagneticField( const KSRootMagneticField& aCopy ) :
            KSComponent(),
            fCurrentField( aCopy.fCurrentField ),
            fCurrentGradient( aCopy.fCurrentGradient ),
            fMagneticFields( aCopy.fMagneticFields )
    {
    }
    KSRootMagneticField* KSRootMagneticField::Clone() const
    {
        return new KSRootMagneticField( *this );
    }
    KSRootMagneticField::~KSRootMagneticField()
    {
    }

    void KSRootMagneticField::CalculateField( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField )
    {
        aField = KThreeVector::sZero;
        for( int tIndex = 0; tIndex < fMagneticFields.End(); tIndex++ )
        {
            fMagneticFields.ElementAt( tIndex )->CalculateField( aSamplePoint, aSampleTime, fCurrentField );
            aField += fCurrentField;
        }
        return;
    }
    void KSRootMagneticField::CalculateGradient( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeMatrix& aGradient )
    {
        aGradient = KThreeMatrix::sZero;
        for( int tIndex = 0; tIndex < fMagneticFields.End(); tIndex++ )
        {
            fMagneticFields.ElementAt( tIndex )->CalculateGradient( aSamplePoint, aSampleTime, fCurrentGradient );
            aGradient += fCurrentGradient;
        }
        return;
    }

    void KSRootMagneticField::CalculateFieldAndGradient( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField, KThreeMatrix& aGradient )
    {
        aField = KThreeVector::sZero;
        aGradient = KThreeMatrix::sZero;
        for( int tIndex = 0; tIndex < fMagneticFields.End(); tIndex++ )
        {
            fMagneticFields.ElementAt( tIndex )->CalculateFieldAndGradient( aSamplePoint, aSampleTime, fCurrentField, fCurrentGradient );
            aField += fCurrentField;
            aGradient += fCurrentGradient;
        }
        return;
    }

    void KSRootMagneticField::AddMagneticField( KSMagneticField* aMagneticField )
    {
        //check that field is not already present
        for( int tIndex = 0; tIndex < fMagneticFields.End(); tIndex++ )
        {
            if(aMagneticField == fMagneticFields.ElementAt( tIndex ) )
            {
                fieldmsg_debug( "<" << GetName() << "> attempted to add magnetic field <" << aMagneticField->GetName() << "> which is already present."  << eom );
                return;
            }
        }

        if( fMagneticFields.AddElement( aMagneticField ) == -1 )
        {
            fieldmsg( eError ) << "<" << GetName() << "> could not add magnetic field <" << aMagneticField->GetName() << ">" << eom;
            return;
        }
        fieldmsg_debug( "<" << GetName() << "> adding magnetic field <" << aMagneticField->GetName() << ">" << eom );
        return;
    }
    void KSRootMagneticField::RemoveMagneticField( KSMagneticField* aMagneticField )
    {
        if( fMagneticFields.RemoveElement( aMagneticField ) == -1 )
        {
            fieldmsg( eError ) << "<" << GetName() << "> could not remove magnetic field <" << aMagneticField->GetName() << ">" << eom;
            return;
        }
        fieldmsg_debug( "<" << GetName() << "> removing magnetic field <" << aMagneticField->GetName() << ">" << eom );
        return;
    }

    STATICINT sKSRootMagneticFieldDict =
        KSDictionary< KSRootMagneticField >::AddCommand( &KSRootMagneticField::AddMagneticField, &KSRootMagneticField::RemoveMagneticField, "add_magnetic_field", "remove_magnetic_field" );

}
