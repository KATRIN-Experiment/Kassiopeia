#include "KSRootWriter.h"
#include "KSWritersMessage.h"

namespace Kassiopeia
{

    KSRootWriter::KSRootWriter() :
        fWriters( 128 )
    {
    }
    KSRootWriter::KSRootWriter( const KSRootWriter& aCopy ) :
        KSComponent(),
        fWriters( aCopy.fWriters )
    {
    }
    KSRootWriter* KSRootWriter::Clone() const
    {
        return new KSRootWriter( *this );
    }
    KSRootWriter::~KSRootWriter()
    {
    }

    void KSRootWriter::AddWriter( KSWriter* aWriter )
    {
        if ( fWriters.AddElement( aWriter ) == -1 )
        {
            wtrmsg( eError ) << "<" << GetName() << "> could not add writer <" << aWriter->GetName() << ">" << eom;
            return;
        }
        wtrmsg_debug( "<" << GetName() << "> adding writer <" << aWriter->GetName() << ">" << eom );
        return;
    }
    void KSRootWriter::RemoveWriter( KSWriter* aWriter )
    {
        if( fWriters.RemoveElement( aWriter ) == -1 )
        {
            wtrmsg( eError ) << "<" << GetName() << "> could not remove writer <" << aWriter->GetName() << ">" << eom;
            return;
        }
        wtrmsg_debug( "<" << GetName() << "> removing writer <" << aWriter->GetName() << ">" << eom );
        return;
    }

    void KSRootWriter::ExecuteRun()
    {
        for( int tIndex = 0; tIndex < fWriters.End(); tIndex++ )
        {
            fWriters.ElementAt( tIndex )->ExecuteRun();
        }
        return;
    }
    void KSRootWriter::ExecuteEvent()
    {
        for( int tIndex = 0; tIndex < fWriters.End(); tIndex++ )
        {
            fWriters.ElementAt( tIndex )->ExecuteEvent();
        }
        return;
    }
    void KSRootWriter::ExecuteTrack()
    {
        for( int tIndex = 0; tIndex < fWriters.End(); tIndex++ )
        {
            fWriters.ElementAt( tIndex )->ExecuteTrack();
        }
        return;
    }
    void KSRootWriter::ExecuteStep()
    {
        for( int tIndex = 0; tIndex < fWriters.End(); tIndex++ )
        {
            fWriters.ElementAt( tIndex )->ExecuteStep();
        }
        return;
    }

    STATICINT sKSWriterDict =
        KSDictionary< KSRootWriter >::AddCommand( &KSRootWriter::AddWriter, &KSRootWriter::RemoveWriter, "add_writer", "remove_writer" );

}

