#ifndef KOMMON_PRINTABLE_COMPONENT_H
#define KOMMON_PRINTABLE_COMPONENT_H

#include <iostream>

namespace katrin{

namespace Kommon{

    class Printable{

    public:
        virtual void Print(std::ostream& output) const = 0;

    protected:
        ~Printable() = default;

    };

    std::ostream& operator<<(std::ostream& output, const Printable& printable);

}

}

#endif
