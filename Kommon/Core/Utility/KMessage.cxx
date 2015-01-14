#include "KMessage.h"

#include <iomanip>
#include <ostream>
using std::cout;
using std::endl;

#include <cstdio>
#include <cstdlib>
#include <execinfo.h>

namespace katrin
{

    KMessage::KMessage( const string& aKey, const string& aDescription, const string& aPrefix, const string& aSuffix ) :
            fKey( aKey ),
            fSystemDescription( aDescription ),
            fSystemPrefix( aPrefix ),
            fSystemSuffix( aSuffix ),

            fErrorColorPrefix( "\33[31;1m" ),
            fErrorColorSuffix( "\33[0m" ),
            fErrorDescription( "ERROR" ),

            fWarningColorPrefix( "\33[33;1m" ),
            fWarningColorSuffix( "\33[0m" ),
            fWarningDescription( "WARNING" ),

            fNormalColorPrefix( "\33[32;1m" ),
            fNormalColorSuffix( "\33[0m" ),
            fNormalDescription( "NORMAL" ),

            fDebugColorPrefix( "\33[36;1m" ),
            fDebugColorSuffix( "\33[0m" ),
            fDebugDescription( "DEBUG" ),

            fDefaultColorPrefix( "\33[37;1m" ),
            fDefaultColorSuffix( "\33[0m" ),
            fDefaultDescription( "UNKNOWN" ),

            fSeverity( eNormal ),

            fColorPrefix( &KMessage::fNormalColorPrefix ),
            fDescription( &KMessage::fNormalDescription ),
            fColorSuffix( &KMessage::fNormalColorSuffix ),

            fMessageLine(),
            fMessageLines(),

            fTerminalVerbosity( KMessageTable::GetInstance()->GetTerminalVerbosity() ),
            fTerminalStream( KMessageTable::GetInstance()->GetTerminalStream() ),
            fLogVerbosity( KMessageTable::GetInstance()->GetLogVerbosity() ),
            fLogStream( KMessageTable::GetInstance()->GetLogStream() )
    {
        fMessageLine.setf( KMessageTable::GetInstance()->GetFormat(), std::ios::floatfield );
        fMessageLine.precision( KMessageTable::GetInstance()->GetPrecision() );
        KMessageTable::GetInstance()->Add( this );
    }
    KMessage::~KMessage()
    {
        KMessageTable::GetInstance()->Remove( this );
    }

    const string& KMessage::GetKey()
    {
        return fKey;
    }
    void KMessage::SetKey( const string& aKey )
    {
        fKey = aKey;
        return;
    }

    void KMessage::SetSeverity( const KMessageSeverity& aSeverity )
    {
        fSeverity = aSeverity;

        switch( fSeverity )
        {
            case eError :
                fColorPrefix = &KMessage::fErrorColorPrefix;
                fDescription = &KMessage::fErrorDescription;
                fColorSuffix = &KMessage::fErrorColorSuffix;
                break;

            case eWarning :
                fColorPrefix = &KMessage::fWarningColorPrefix;
                fDescription = &KMessage::fWarningDescription;
                fColorSuffix = &KMessage::fWarningColorSuffix;
                break;

            case eNormal :
                fColorPrefix = &KMessage::fNormalColorPrefix;
                fDescription = &KMessage::fNormalDescription;
                fColorSuffix = &KMessage::fNormalColorSuffix;
                break;

            case eDebug :
                fColorPrefix = &KMessage::fDebugColorPrefix;
                fDescription = &KMessage::fDebugDescription;
                fColorSuffix = &KMessage::fDebugColorSuffix;
                break;

            default :
                fColorPrefix = &KMessage::fDebugColorPrefix;
                fDescription = &KMessage::fDebugDescription;
                fColorSuffix = &KMessage::fDebugColorSuffix;
                break;
        }

        return;
    }
    void KMessage::Flush()
    {
        if( (fSeverity <= fTerminalVerbosity) && (fTerminalStream != NULL) && (fTerminalStream->good() == true) )
        {
            for( vector< pair< string, char > >::iterator It = fMessageLines.begin(); It != fMessageLines.end(); It++ )
            {
                (*fTerminalStream) << this->*fColorPrefix << fSystemPrefix << "[" << fSystemDescription << " " << this->*fDescription << " MESSAGE] " << It->first << fSystemSuffix << this->*fColorSuffix << It->second;
            }
            (*fTerminalStream).flush();
        }

        if( (fSeverity <= fLogVerbosity) && (fLogStream != NULL) && (fLogStream->good() == true) )
        {
            for( vector< pair< string, char > >::iterator It = fMessageLines.begin(); It != fMessageLines.end(); It++ )
            {
                (*fLogStream) << fSystemPrefix << "[" << fSystemDescription << " " << this->*fDescription << " MESSAGE] " << It->first << fSystemSuffix << "\n";
            }
            (*fLogStream).flush();
        }

        while( !fMessageLines.empty() )
        {
            fMessageLines.pop_back();
        }

        if( fSeverity == eError )
        {
            Shutdown();
        }

        return;
    }

    void KMessage::Shutdown()
    {
        const size_t MaxFrameCount = 512;
        void* FrameArray[ MaxFrameCount ];
        const size_t FrameCount = backtrace( FrameArray, MaxFrameCount );
        char** FrameSymbols = backtrace_symbols( FrameArray, FrameCount );

        if( (fSeverity <= fTerminalVerbosity) && (fTerminalStream != NULL) && (fTerminalStream->good() == true) )
        {
            (*fTerminalStream) << this->*fColorPrefix << fSystemPrefix << "[" << fSystemDescription << " " << this->*fDescription << " MESSAGE] shutting down..." << fSystemSuffix << this->*fColorSuffix << '\n';
            (*fTerminalStream) << this->*fColorPrefix << fSystemPrefix << "[" << fSystemDescription << " " << this->*fDescription << " MESSAGE] stack trace:" << fSystemSuffix << this->*fColorSuffix << '\n';
            for( size_t Index = 0; Index < FrameCount; Index++ )
            {
                (*fTerminalStream) << this->*fColorPrefix << fSystemPrefix << "[" << fSystemDescription << " " << this->*fDescription << " MESSAGE] " << FrameSymbols[ Index ] << fSystemSuffix << this->*fColorSuffix << '\n';
            }
            (*fTerminalStream).flush();
        }

        if( (fSeverity <= fLogVerbosity) && (fLogStream != NULL) && (fLogStream->good() == true) )
        {
            (*fLogStream) << fSystemPrefix << "[" << fSystemDescription << " " << this->*fDescription << " MESSAGE] shutting down..." << fSystemSuffix << '\n';
            (*fLogStream) << fSystemPrefix << "[" << fSystemDescription << " " << this->*fDescription << " MESSAGE] stack trace:" << fSystemSuffix << '\n';
            for( size_t Index = 0; Index < FrameCount; Index++ )
            {
                (*fLogStream) << fSystemPrefix << "[" << fSystemDescription << " " << this->*fDescription << " MESSAGE] " << FrameSymbols[ Index ] << fSystemSuffix << '\n';
            }
            (*fLogStream).flush();
        }

        free( FrameSymbols );
        exit( -1 );

        return;
    }

    void KMessage::SetFormat( const KMessageFormat& aFormat )
    {
        fMessageLine.setf( aFormat, std::ios::floatfield );
        return;
    }
    void KMessage::SetPrecision( const KMessagePrecision& aPrecision )
    {
        fMessageLine.precision( aPrecision );
        return;
    }
    void KMessage::SetTerminalVerbosity( const KMessageSeverity& aVerbosity )
    {
        fTerminalVerbosity = aVerbosity;
        return;
    }
    void KMessage::SetTerminalStream( ostream* aTerminalStream )
    {
        fTerminalStream = aTerminalStream;
        return;
    }
    void KMessage::SetLogVerbosity( const KMessageSeverity& aVerbosity )
    {
        fLogVerbosity = aVerbosity;
        return;
    }
    void KMessage::SetLogStream( ostream* aLogStream )
    {
        fLogStream = aLogStream;
        return;
    }

}

namespace katrin
{

    KMessageTable::KMessageTable() :
            fMessageMap(),
            fFormat( cout.flags() ),
            fPrecision( cout.precision() ),
            fTerminalVerbosity( eNormal ),
            fTerminalStream( &cout ),
            fLogVerbosity( eNormal ),
            fLogStream( NULL )
    {
    }
    KMessageTable::~KMessageTable()
    {
    }

    //********
    //messages
    //********

    void KMessageTable::Add( KMessage* aMessage )
    {
        MessageIt tIter = fMessageMap.find( aMessage->GetKey() );
        if( tIter == fMessageMap.end() )
        {
            fMessageMap.insert( MessageEntry( aMessage->GetKey(), aMessage ) );
        }
        return;
    }
    KMessage* KMessageTable::Get( const string& aKey )
    {
        MessageIt tIter = fMessageMap.find( aKey );
        if( tIter != fMessageMap.end() )
        {
            return tIter->second;
        }
        return NULL;
    }
    void KMessageTable::Remove( KMessage* aMessage )
    {
        MessageIt tIter = fMessageMap.find( aMessage->GetKey() );
        if( tIter != fMessageMap.end() )
        {
            fMessageMap.erase( tIter );
        }
        return;
    }

    void KMessageTable::SetFormat( const KMessageFormat& aFormat )
    {
        fFormat = aFormat;
        MessageIt tIter;
        for( tIter = fMessageMap.begin(); tIter != fMessageMap.end(); tIter++ )
        {
            tIter->second->SetFormat( fFormat );
        }
        return;
    }
    const KMessageFormat& KMessageTable::GetFormat()
    {
        return fFormat;
    }

    void KMessageTable::SetPrecision( const KMessagePrecision& aPrecision )
    {
        fPrecision = aPrecision;
        MessageIt tIter;
        for( tIter = fMessageMap.begin(); tIter != fMessageMap.end(); tIter++ )
        {
            tIter->second->SetPrecision( fPrecision );
        }
        return;
    }
    const KMessagePrecision& KMessageTable::GetPrecision()
    {
        return fPrecision;
    }

    void KMessageTable::SetTerminalVerbosity( const KMessageSeverity& aVerbosity )
    {
        fTerminalVerbosity = aVerbosity;
        MessageIt tIter;
        for( tIter = fMessageMap.begin(); tIter != fMessageMap.end(); tIter++ )
        {
            tIter->second->SetTerminalVerbosity( fTerminalVerbosity );
        }
        return;
    }
    const KMessageSeverity& KMessageTable::GetTerminalVerbosity()
    {
        return fTerminalVerbosity;
    }

    void KMessageTable::SetTerminalStream( ostream* aTerminalStream )
    {
        fTerminalStream = aTerminalStream;
        MessageIt tIter;
        for( tIter = fMessageMap.begin(); tIter != fMessageMap.end(); tIter++ )
        {
            tIter->second->SetTerminalStream( fTerminalStream );
        }
        return;
    }
    ostream* KMessageTable::GetTerminalStream()
    {
        return fTerminalStream;
    }

    void KMessageTable::SetLogVerbosity( const KMessageSeverity& aVerbosity )
    {
        fLogVerbosity = aVerbosity;
        MessageIt tIter;
        for( tIter = fMessageMap.begin(); tIter != fMessageMap.end(); tIter++ )
        {
            tIter->second->SetLogVerbosity( fLogVerbosity );
        }
        return;
    }
    const KMessageSeverity& KMessageTable::GetLogVerbosity()
    {
        return fLogVerbosity;
    }

    void KMessageTable::SetLogStream( ostream* aLogStream )
    {
        fLogStream = aLogStream;
        MessageIt tIter;
        for( tIter = fMessageMap.begin(); tIter != fMessageMap.end(); tIter++ )
        {
            tIter->second->SetLogStream( fLogStream );
        }
        return;
    }
    ostream* KMessageTable::GetLogStream()
    {
        return fLogStream;
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

#include "KElementProcessor.hh"

namespace katrin
{

    KMessageData::KMessageData() :
            fKey( "none" ),
            fFormat( cout.flags() ),
            fPrecision( cout.precision() ),
            fTerminalVerbosity( eNormal ),
            fLogVerbosity( eNormal )
    {
    }
    KMessageData::~KMessageData()
    {
    }

    template< >
    KMessageDataBuilder::~KComplexElement()
    {
    }

    static int sKMessageDataStructure = KMessageDataBuilder::Attribute< string >( "key" ) + KMessageDataBuilder::Attribute< string >( "terminal" ) + KMessageDataBuilder::Attribute< string >( "log" ) + KMessageDataBuilder::Attribute< string >( "format" ) + KMessageDataBuilder::Attribute< KMessagePrecision >( "precision" );

    template< >
    KMessageTableBuilder::~KComplexElement()
    {
    }

    static int sKMessageTableStructure = KMessageTableBuilder::Attribute< string >( "terminal" ) + KMessageTableBuilder::Attribute< string >( "log" ) + KMessageTableBuilder::Attribute< string >( "format" ) + KMessageTableBuilder::Attribute< KMessagePrecision >( "precision" ) + KMessageTableBuilder::ComplexElement< KTextFile >( "file" ) + KMessageTableBuilder::ComplexElement< KMessageData >( "message" );

    static int sMessageTable = KElementProcessor::ComplexElement< KMessageTable >( "messages" );

}
