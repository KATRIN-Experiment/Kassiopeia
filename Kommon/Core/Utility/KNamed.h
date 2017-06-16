#ifndef KNAMED_H_
#define KNAMED_H_

#include <string>
#include <ostream>

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
            bool HasName( const std::string& aName ) const;
            const std::string& GetName() const;
            void SetName( const std::string& aName );

        protected:
            std::string fName;
    };

    inline bool KNamed::HasName( const std::string& aName ) const
    {
        if( fName == aName )
        {
            return true;
        }
        return false;
    }
    inline void KNamed::SetName( const std::string& aName )
    {
        fName = aName;
        return;
    }
    inline const std::string& KNamed::GetName() const
    {
        return fName;
    }

    inline std::ostream& operator<<( std::ostream& aStream, const KNamed& aNamed )
    {
        aStream << "<" << aNamed.GetName() << ">";
        return aStream;
    }

}

#endif
