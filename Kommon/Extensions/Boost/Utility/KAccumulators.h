/**
 * @file KAccumulators.h
 *
 * @date 08.01.2014
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 *
 *
 * This file contains accumulator definitions for the boost accumulator framework:
 * http://www.boost.org/doc/libs/1_55_0/doc/html/accumulators/user_s_guide.html
 *
 */
#ifndef KACCUMULATORS_H_
#define KACCUMULATORS_H_

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

namespace boost { namespace accumulators {

// Standard Deviation
namespace impl {
    template<typename Sample>
    struct error_accumulator : accumulator_base
    {
        typedef Sample result_type;
        error_accumulator(dont_care) {}

        template<typename Args>
        result_type result(Args const &args) const
        {
            return std::sqrt( std::max<result_type>(0.0, variance(args[accumulator])) );
        }
    };
}

namespace tag {
    struct error : depends_on< variance >
    {
        typedef boost::accumulators::impl::error_accumulator< mpl::_1 > impl;
    };
}

namespace extract {
    extractor< tag::error > const error = {};
}

// Unbiased Variance
namespace impl {
    template<typename Sample>
    struct variance_unbiased_accumulator : accumulator_base
    {
        typedef Sample result_type;
        variance_unbiased_accumulator(dont_care) {}

        template<typename Args>
        result_type result(Args const &args) const
        {
            const size_t N = extract::count(args[accumulator]);
            if (N < 2)
                return std::numeric_limits<result_type>::quiet_NaN();
            return numeric::average(variance(args[accumulator]) * (result_type) N, (N-1) ) ;
        }
    };
}

namespace tag {
    struct variance_unbiased : depends_on< variance >
    {
        typedef boost::accumulators::impl::variance_unbiased_accumulator< mpl::_1 > impl;
    };
}

namespace extract {
    extractor< tag::variance_unbiased > const variance_unbiased = {};
}

namespace impl {
    template<typename Sample>
    struct error_unbiased_accumulator : accumulator_base
    {
        typedef Sample result_type;
        error_unbiased_accumulator(dont_care) {}

        template<typename Args>
        result_type result(Args const &args) const
        {
            const size_t N = extract::count(args[accumulator]);
            if (N < 2)
                return std::numeric_limits<result_type>::quiet_NaN();
            else
                return std::sqrt( std::max<result_type>(0.0, extract::variance_unbiased(args[accumulator])) );
        }
    };
}

namespace tag {
    struct error_unbiased : depends_on< variance_unbiased >
    {
        typedef boost::accumulators::impl::error_unbiased_accumulator< mpl::_1 > impl;
    };
}

namespace extract {
    extractor< tag::error_unbiased > const error_unbiased = {};
}

namespace impl {
    template<typename Sample>
    struct error_of_mean_accumulator : accumulator_base
    {
        typedef Sample result_type;
        error_of_mean_accumulator(dont_care) {}

        template<typename Args>
        result_type result(Args const &args) const
        {
            const size_t N = extract::count(args[accumulator]);
            if (N < 2)
                return std::numeric_limits<result_type>::quiet_NaN();
            else
                return std::sqrt( numeric::average( std::max<result_type>(0.0, extract::variance( args[accumulator] )), (N - 1)) );
        }
    };
}

namespace tag {
    struct error_of_mean : depends_on< variance >
    {
        typedef boost::accumulators::impl::error_of_mean_accumulator< mpl::_1 > impl;
    };
}

namespace extract {
    extractor< tag::error_of_mean > const error_of_mean = {};
}


// pulling the extractors into the boost::accumulators namespace:

using extract::variance_unbiased;
using extract::error;
using extract::error_unbiased;
using extract::error_of_mean;

} }


#endif /* KACCUMULATORS_H_ */
