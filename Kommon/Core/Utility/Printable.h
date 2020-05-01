/**
 * @file Printable.h
 * @brief Inherit from this class to define operator<< using your virtual Print(stream)
 * @date 17.11.2018
 * @author Valerian Sibille <vsibille@mit.edu>
 */

#ifndef KOMMON_PRINTABLE_COMPONENT_H
#define KOMMON_PRINTABLE_COMPONENT_H

#include <iostream>

namespace katrin
{

namespace Kommon
{

class Printable
{

  public:
    virtual void Print(std::ostream& output) const = 0;

  protected:
    ~Printable() = default;
};

std::ostream& operator<<(std::ostream& output, const Printable& printable);

}  // namespace Kommon

}  // namespace katrin

#endif
