#ifndef KNAMED_H_
#define KNAMED_H_

#include <string>
using std::string;

#include <ostream>
using std::ostream;

namespace katrin
{

    class KNamed
    {
        public:
            KNamed();
            KNamed( const KNamed& aNamed );
            virtual ~KNamed();

            //**************
            //identification
            //**************

        public:
            bool HasName( const string& aName ) const;
            const string& GetName() const;
            void SetName( const string& aName );

        protected:
            string fName;
    };

    inline bool KNamed::HasName( const string& aName ) const
    {
        if( fName == aName )
        {
            return true;
        }
        return false;
    }
    inline void KNamed::SetName( const string& aName )
    {
        fName = aName;
        return;
    }
    inline const string& KNamed::GetName() const
    {
        return fName;
    }

    inline ostream& operator<<( ostream& aStream, const KNamed& aNamed )
    {
        aStream << "<" << aNamed.GetName() << ">";
        return aStream;
    }

}

#endif
