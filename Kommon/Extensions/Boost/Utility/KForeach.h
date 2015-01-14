/**
 * @file KForeach.h
 *
 * @date 16.06.2013
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */

#ifndef KFOREACH_H_
#define KFOREACH_H_

#ifdef __CDT_PARSER__
    #define foreach(a, b) for(a : b)
#else
    #include <boost/foreach.hpp>
    #define foreach(a, b) BOOST_FOREACH(a, b)
#endif

#endif /* KFOREACH_H_ */
