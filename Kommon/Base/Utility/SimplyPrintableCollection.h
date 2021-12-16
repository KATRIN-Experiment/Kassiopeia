/**
 * @file SimplyPrintableCollection.h
 * @brief Inherit from this class to make your collection (begin and end available) printable line by line
 * @date 09.03.2019
 * @author Valerian Sibille <vsibille@mit.edu>
 */

#ifndef KOMMON_SIMPLYPRINTABLECOLLECTION_H
#define KOMMON_SIMPLYPRINTABLECOLLECTION_H

#include "OstreamJoiner.h"

#include <iostream>

namespace katrin
{

namespace Kommon
{

template<class Derived> struct SimplyPrintableCollection
{

    const Derived& derived() const;
};


template<class Derived>
std::ostream& operator<<(std::ostream& output, const SimplyPrintableCollection<Derived>& printableCollection);

template<class Derived> const Derived& SimplyPrintableCollection<Derived>::derived() const
{

    return static_cast<const Derived&>(*this);
}

template<class Derived>
std::ostream& operator<<(std::ostream& output, const SimplyPrintableCollection<Derived>& printableCollection)
{

    std::copy(std::begin(printableCollection.derived()),
              std::end(printableCollection.derived()),
              MakeOstreamJoiner(output, "\n"));
    return output;
}

} /* namespace Kommon */

}  // namespace katrin

#endif
