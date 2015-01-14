#include "KToken.hh"

#include <cstdlib>

namespace katrin
{

    KToken::KToken() :
        fValue( "" ),
        fPath( "" ),
        fFile( "" ),
        fLine( 0 ),
        fColumn( 0 )
    {
    }
    KToken::KToken( const KToken& aToken ) :
        fValue( aToken.fValue ),
        fPath( aToken.fPath ),
        fFile( aToken.fFile ),
        fLine( aToken.fLine ),
        fColumn( aToken.fColumn )
    {
    }
    KToken::~KToken()
    {
    }

    void KToken::SetValue( const string& aValue )
    {
        fValue = aValue;
        return;
    }

    const string& KToken::GetValue() const
    {
        return fValue;
    }

    void KToken::SetPath( const string& aPath )
    {
        fPath = aPath;
        return;
    }
    const string& KToken::GetPath() const
    {
        return fPath;
    }

    void KToken::SetFile( const string& aFile )
    {
        fFile = aFile;
        return;
    }
    const string& KToken::GetFile() const
    {
        return fFile;
    }

    void KToken::SetLine( const int& aLine )
    {
        fLine = aLine;
        return;
    }
    const int& KToken::GetLine() const
    {
        return fLine;
    }

    void KToken::SetColumn( const int& aColumn )
    {
        fColumn = aColumn;
        return;
    }
    const int& KToken::GetColumn() const
    {
        return fColumn;
    }

}
