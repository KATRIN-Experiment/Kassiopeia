#include "KSReadObjectROOT.h"

namespace Kassiopeia
{

    KSReadObjectROOT::KSReadObjectROOT( TTree* aStructureTree, TTree* aPresenceTree, TTree* aDataTree ) :
            fPresences(),
            fValid( false ),
            fIndex( 0 ),
            fStructure( aStructureTree ),
            fPresence( aPresenceTree ),
            fData( aDataTree )
    {
        string tLabel;
        string* tLabelPointer = &tLabel;
        string** tLabelHandle = &tLabelPointer;
        fStructure->SetBranchAddress( "LABEL", tLabelHandle );

        string tType;
        string* tTypePointer = &tType;
        string** tTypeHandle = &tTypePointer;
        fStructure->SetBranchAddress( "TYPE", tTypeHandle );

        for( Long64_t tStructureIndex = 0; tStructureIndex < fStructure->GetEntries(); tStructureIndex++ )
        {
            fStructure->GetEntry( tStructureIndex );

            readermsg_debug( "analyzing structure with label <" << tLabel << "> and type <" << tType << ">" << eom );

            if( tType == string( "bool" ) )
            {
                fData->SetBranchAddress( tLabel.c_str(), Add< KSBool >( tLabel ).Pointer() );
                continue;
            }

            if( tType == string( "unsigned_char" ) )
            {
                fData->SetBranchAddress( tLabel.c_str(), Add< KSUChar >( tLabel ).Pointer() );
                continue;
            }
            if( tType == string( "char" ) )
            {
                fData->SetBranchAddress( tLabel.c_str(), Add< KSChar >( tLabel ).Pointer() );
                continue;
            }

            if( tType == string( "unsigned_short" ) )
            {
                fData->SetBranchAddress( tLabel.c_str(), Add< KSUShort >( tLabel ).Pointer() );
                continue;
            }
            if( tType == string( "short" ) )
            {
                fData->SetBranchAddress( tLabel.c_str(), Add< KSShort >( tLabel ).Pointer() );
                continue;
            }

            if( tType == string( "unsigned_int" ) )
            {
                fData->SetBranchAddress( tLabel.c_str(), Add< KSUInt >( tLabel ).Pointer() );
                continue;
            }
            if( tType == string( "int" ) )
            {
                fData->SetBranchAddress( tLabel.c_str(), Add< KSInt >( tLabel ).Pointer() );
                continue;
            }

            if( tType == string( "unsigned_long" ) )
            {
                fData->SetBranchAddress( tLabel.c_str(), Add< KSULong >( tLabel ).Pointer() );
                continue;
            }
            if( tType == string( "long" ) )
            {
                fData->SetBranchAddress( tLabel.c_str(), Add< KSLong >( tLabel ).Pointer() );
                continue;
            }

            if( tType == string( "float" ) )
            {
                fData->SetBranchAddress( tLabel.c_str(), Add< KSFloat >( tLabel ).Pointer() );
                continue;
            }
            if( tType == string( "double" ) )
            {
                fData->SetBranchAddress( tLabel.c_str(), Add< KSDouble >( tLabel ).Pointer() );
                continue;
            }

            if( tType == string( "string" ) )
            {
                fData->SetBranchAddress( tLabel.c_str(), Add< KSString >( tLabel ).Handle() );
                continue;
            }

            if( tType == string( "two_vector" ) )
            {
                KSTwoVector& tTwoVector = Add< KSTwoVector >( tLabel );
                fData->SetBranchAddress( (tLabel + string( "_x" )).c_str(), &(tTwoVector.Value().X()) );
                fData->SetBranchAddress( (tLabel + string( "_y" )).c_str(), &(tTwoVector.Value().Y()) );
                continue;
            }
            if( tType == string( "three_vector" ) )
            {
                KSThreeVector& tTwoVector = Add< KSThreeVector >( tLabel );
                fData->SetBranchAddress( (tLabel + string( "_x" )).c_str(), &(tTwoVector.Value().X()) );
                fData->SetBranchAddress( (tLabel + string( "_y" )).c_str(), &(tTwoVector.Value().Y()) );
                fData->SetBranchAddress( (tLabel + string( "_z" )).c_str(), &(tTwoVector.Value().Z()) );
                continue;
            }

            readermsg( eError ) << "could not analyze branch with label <" << tLabel << "> and type <" << tType << ">" << eom;
        }

        unsigned int tIndex;
        unsigned int* tIndexPointer = &tIndex;
        fPresence->SetBranchAddress( "INDEX", tIndexPointer );

        unsigned int tLength;
        unsigned int* tLengthPointer = &tLength;
        fPresence->SetBranchAddress( "LENGTH", tLengthPointer );

        vector< Presence > tPresences;
        unsigned int tEntry = 0;
        for( Long64_t tPresenceIndex = 0; tPresenceIndex < fPresence->GetEntries(); tPresenceIndex++ )
        {
            fPresence->GetEntry( tPresenceIndex );

            readermsg_debug( "analyzing presence with index <" << tIndex << "> and length <" << tLength << ">" << eom );

            tPresences.push_back( Presence( tIndex, tLength, tEntry ) );
            tEntry += tLength;
        }

        //rearrange presence data to avoid exponential grow of analysis time
        unsigned int tFirstIndex;
        unsigned int tLastIndex;
        unsigned int tFirstEntry;
        unsigned int tTotalLength = 0;
        for( vector< Presence >::iterator tIt = tPresences.begin(); tIt != tPresences.end(); tIt++ )
        {
        	tIndex = tIt->fIndex;
        	tLength = tIt->fLength;
        	tEntry = tIt->fEntry;

        	if ( tIt == tPresences.begin() )
        	{
        		tFirstIndex = tIndex;
        		tLastIndex = tIndex;
        		tFirstEntry = tEntry;
        		tTotalLength= tLength;
        		continue;
        	}

        	if ( tIndex == tLastIndex + 1 )
        	{
        		tTotalLength += tLength;
        	}
        	else
        	{
        		fPresences.push_back( Presence( tFirstIndex, tTotalLength, tFirstEntry ) );
        		tFirstIndex = tIndex;
        		tFirstEntry = tEntry;
        		tTotalLength = tLength;
        	}
			tLastIndex = tIndex;

        }
		fPresences.push_back( Presence( tFirstIndex, tTotalLength, tFirstEntry ) );

    }

    KSReadObjectROOT::~KSReadObjectROOT()
    {
    }

    void KSReadObjectROOT::operator++( int )
    {
        fIndex++;
        for( vector< Presence >::iterator tIt = fPresences.begin(); tIt != fPresences.end(); tIt++ )
        {
            if( tIt->fIndex > fIndex )
            {
                fValid = false;
                return;
            }
            if( tIt->fIndex + tIt->fLength > fIndex )
            {
                fValid = true;
                fData->GetEntry( tIt->fEntry + (fIndex - tIt->fIndex) );
                return;
            }
        }
        fValid = false;
        return;
    }
    void KSReadObjectROOT::operator--( int )
    {
        fIndex--;
        for( vector< Presence >::iterator tIt = fPresences.begin(); tIt != fPresences.end(); tIt++ )
        {
            if( tIt->fIndex > fIndex )
            {
                fValid = false;
                return;
            }
            if( tIt->fIndex + tIt->fLength > fIndex )
            {
                fValid = true;
                fData->GetEntry( tIt->fEntry + (fIndex - tIt->fIndex) );
                return;
            }
        }
        fValid = false;
        return;
    }
    void KSReadObjectROOT::operator<<( const unsigned int& aValue )
    {
        fIndex = aValue;
        for( vector< Presence >::iterator tIt = fPresences.begin(); tIt != fPresences.end(); tIt++ )
        {
            if( tIt->fIndex > fIndex )
            {
                fValid = false;
                return;
            }
            if( tIt->fIndex + tIt->fLength > fIndex )
            {
                fValid = true;
                fData->GetEntry( tIt->fEntry + (fIndex - tIt->fIndex) );
                return;
            }
        }
        fValid = false;
        return;
    }

    bool KSReadObjectROOT::Valid() const
    {
        return fValid;
    }
    unsigned int KSReadObjectROOT::Index() const
    {
        return fIndex;
    }
    bool KSReadObjectROOT::operator<( const unsigned int& aValue ) const
    {
        return (fIndex < aValue);
    }
    bool KSReadObjectROOT::operator<=( const unsigned int& aValue ) const
    {
        return (fIndex <= aValue);
    }
    bool KSReadObjectROOT::operator>( const unsigned int& aValue ) const
    {
        return (fIndex > aValue);
    }
    bool KSReadObjectROOT::operator>=( const unsigned int& aValue ) const
    {
        return (fIndex >= aValue);
    }
    bool KSReadObjectROOT::operator==( const unsigned int& aValue ) const
    {
        return (fIndex == aValue);
    }
    bool KSReadObjectROOT::operator!=( const unsigned int& aValue ) const
    {
        return (fIndex != aValue);
    }

}
