#ifndef Kommon_KToken_hh_
#define Kommon_KToken_hh_

#include "KInitializationMessage.hh"

#include <string>
using std::string;

#include <sstream>
using std::istringstream;

#include <cstdlib>

namespace katrin
{

    class KToken
    {
        public:
            KToken();
            KToken( const KToken& aToken );
            virtual ~KToken();

            virtual KToken* Clone() = 0;

        public:
            void SetValue( const string& aValue );
            const string& GetValue() const;

            template< typename XDataType >
            XDataType GetValue() const;

            void SetPath( const string& aPath );
            const string& GetPath() const;

            void SetFile( const string& aFile );
            const string& GetFile() const;

            void SetLine( const int& aLine );
            const int& GetLine() const;

            void SetColumn( const int& aColumn );
            const int& GetColumn() const;

        private:
            string fValue;

            string fPath;
            string fFile;
            int fLine;
            int fColumn;
    };

    template< typename XDataType >
    inline XDataType KToken::GetValue() const
    {
        istringstream Converter( fValue );
        XDataType Data;
        Converter >> Data;
        if (Converter.fail() || (Data != Data) )  // also check for NaN
        {
            string TypeName = KMessage::TypeName< XDataType >();
            initmsg( eWarning ) << "error processing token <" << fValue << "> with type <" << TypeName << ">, replaced with <" << Data << ">" << ret;
            initmsg( eWarning ) << "in path <" << fPath << "> in file <" << fFile << "> at line <" << fLine << "> at column <" << fColumn << ">" << eom;
        }
        return Data;
    }

    template<>
    inline bool KToken::GetValue< bool >() const
    {
        if ( fValue == string("0")
                || fValue == string("false") || fValue == string("False") || fValue == string("FALSE")
                || fValue == string("no") || fValue == string("No") || fValue == string("NO") )
        {
            return false;
        }
        return true;
    }

    template<>
    inline string KToken::GetValue< string >() const
    {
        return fValue;
    }

}

#endif
