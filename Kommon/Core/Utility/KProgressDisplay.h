#ifndef _katrin_KProgressDisplay_h_
#define _katrin_KProgressDisplay_h_

/*
 * Display progress indicator on the terminal, like this:
 *   [################                                  ] 25.0%
 *
 * Usage:
 *   KProgressDisplay progress(256);
 *   progress.start();
 *   while (progress.count() < 256) { ++progress; }
 */

#include <ostream>
#include <iomanip>

namespace katrin {

// re-implemented from boost::progress_display
class KProgressDisplay
{
public:
    explicit KProgressDisplay( unsigned long expected_count, std::ostream& os = std::cout )
    : fStream(os), fExpectedCount(expected_count)
    {
        fCount = fNextTicCount = 0;
    }

    void start( unsigned long initial = 0 )
    {
        fNextTicCount = 0;
        fCount = initial;

        if ( ! fExpectedCount )
            fExpectedCount = 1;  // prevent divide by zero

        display_tic();
    }

    unsigned long operator+=( unsigned long increment )
    {
        if ( (fCount += increment) >= fNextTicCount )
            display_tic();

        return fCount;
    }

    unsigned long  operator++()         { return operator+=( 1 ); }
    unsigned long  count() const        { return fCount; }
    unsigned long  expected() const     { return fExpectedCount; }

protected:
    void display_tic()
    {
        // use of floating point ensures that both large and small counts
        // work correctly.  static_cast<>() is also used several places
        // to suppress spurious compiler warnings.
        double progress = static_cast<double>(fCount) / fExpectedCount;
        unsigned int tics_needed = static_cast<unsigned int>( progress * 50. );

        fStream << '[';
        for ( unsigned int tic = 0; tic < 50; ++tic )
            fStream << (tic < tics_needed ? '#' : ' ');
        fStream << ']'
                << ' ' << std::fixed << std::setprecision(1)
                << static_cast<double>( static_cast<int>( progress * 1000. ) / 10. ) << '%'
                << '\r' << std::flush;

        if ( fCount == fExpectedCount )
            fStream << std::endl;

        fNextTicCount = static_cast<unsigned long>( (tics_needed / 50.) * fExpectedCount );
    }

private:
    std::ostream& fStream;

    unsigned long fCount, fExpectedCount, fNextTicCount;
};

}

#endif /* */
