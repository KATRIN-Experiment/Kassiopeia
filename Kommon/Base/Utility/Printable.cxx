#include "Printable.h"

namespace katrin
{

namespace Kommon
{


std::ostream& operator<<(std::ostream& output, const Printable& printable)
{

    printable.Print(output);
    return output;
}

}  // namespace Kommon

}  // namespace katrin
