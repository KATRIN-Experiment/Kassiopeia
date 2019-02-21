#ifndef KNAMED_H_
#define KNAMED_H_

#include <string>
#include <ostream>
#include "Printable.h"

namespace katrin
{

    class KNamed : public Kommon::Printable
    {
        public:
            KNamed();
            KNamed( const KNamed& aNamed );
            virtual ~KNamed() = default;
            bool HasName( const std::string& aName ) const;
            const std::string& GetName() const;
            void SetName(std::string aName);
            void Print(std::ostream& output) const;

        private:
            std::string fName;
    };

}

#endif
