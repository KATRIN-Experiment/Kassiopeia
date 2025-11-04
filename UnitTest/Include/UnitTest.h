#ifndef UNITTEST_H_
#define UNITTEST_H_

#include "doctest.h"
#include <csignal>

/* Some useful macros to access values from numeric_limits */

#include <limits>
#include <string>

#define EPSILON_FLOAT  (std::numeric_limits<float>::epsilon())
#define EPSILON_DOUBLE (std::numeric_limits<double>::epsilon())

#define ROUND_ERROR_FLOAT  (std::numeric_limits<float>::round_error())
#define ROUND_ERROR_DOUBLE (std::numeric_limits<double>::round_error())

#define MIN_FLOAT  (std::numeric_limits<float>::min())
#define MIN_DOUBLE (std::numeric_limits<double>::min())

#define MAX_FLOAT  (std::numeric_limits<float>::max())
#define MAX_DOUBLE (std::numeric_limits<double>::max())


/* Some additional macros for certain value checks in unit tests */

// Map GoogleTest macros to doctest macros for compatibility
#define EXPECT_TRUE(cond) CHECK(cond)
#define EXPECT_FALSE(cond) CHECK(!(cond))
#define EXPECT_EQ(a, b) CHECK((a) == (b))
#define EXPECT_NE(a, b) CHECK((a) != (b))
#define EXPECT_LT(a, b) CHECK((a) < (b))
#define EXPECT_LE(a, b) CHECK((a) <= (b))
#define EXPECT_GT(a, b) CHECK((a) > (b))
#define EXPECT_GE(a, b) CHECK((a) >= (b))
#define EXPECT_NEAR(a, b, tol) CHECK((a) == doctest::Approx(b).epsilon(tol))
#define EXPECT_DOUBLE_EQ(a, b) CHECK((a) == doctest::Approx(b))
#define EXPECT_FLOAT_EQ(a, b) CHECK((a) == doctest::Approx(b))
#define EXPECT_DOUBLE_LT(a, b) CHECK((a) < (b))

#define ASSERT_TRUE(cond) REQUIRE(cond)
#define ASSERT_FALSE(cond) REQUIRE(!(cond))
#define ASSERT_EQ(a, b) REQUIRE((a) == (b))
#define ASSERT_NE(a, b) REQUIRE((a) != (b))
#define ASSERT_LT(a, b) REQUIRE((a) < (b))
#define ASSERT_LE(a, b) REQUIRE((a) <= (b))
#define ASSERT_GT(a, b) REQUIRE((a) > (b))
#define ASSERT_GE(a, b) REQUIRE((a) >= (b))
#define ASSERT_NEAR(a, b, tol) REQUIRE((a) == doctest::Approx(b).epsilon(tol))
#define ASSERT_DOUBLE_EQ(a, b) REQUIRE((a) == doctest::Approx(b))
#define ASSERT_FLOAT_EQ(a, b) REQUIRE((a) == doctest::Approx(b))
#define ASSERT_DOUBLE_LT(a, b) REQUIRE((a) < (b))

#define EXPECT_THROW(expr, exception_type) CHECK_THROWS_AS(expr, exception_type)
#define ASSERT_THROW(expr, exception_type) REQUIRE_THROWS_AS(expr, exception_type)

#define EXPECT_PTR(p) CHECK((p) != NULL)
#define ASSERT_PTR(p) REQUIRE((p) != NULL)

#define EXPECT_NULL(p) CHECK((p) == NULL)
#define ASSERT_NULL(p) REQUIRE((p) == NULL)

#define EXPECT_STRING_EQ(a, b) CHECK(std::string(a) == std::string(b))
#define ASSERT_STRING_EQ(a, b) REQUIRE(std::string(a) == std::string(b))

#define EXPECT_VECTOR_NULL(a) CHECK((a).Magnitude() < ROUND_ERROR_DOUBLE)
#define ASSERT_VECTOR_NULL(a) REQUIRE((a).Magnitude() < ROUND_ERROR_DOUBLE)

#define EXPECT_VECTOR_NEAR(a, b) CHECK(((a) - (b)).Magnitude() < ROUND_ERROR_DOUBLE)
#define ASSERT_VECTOR_NEAR(a, b) REQUIRE(((a) - (b)).Magnitude() < ROUND_ERROR_DOUBLE)


/**
 * A fancy fixture class which adds timeouts to fixtures/tests deriving from this one.
 * @author J. Behrens
 * 
 * Note: Doctest doesn't use test fixtures the same way as GoogleTest.
 * This class is kept for backward compatibility, but tests should be converted
 * to use doctest's TEST_CASE and SUBCASE instead of TEST_F.
 * Timeout functionality would need to be implemented differently with doctest.
 */
class TimeoutTest
{
  public:
    static int GetTimeoutSeconds()
    {
        return 60;
    }  // all tests should complete within 60 seconds

  protected:
    typedef void Sigfunc(int);

    static Sigfunc* signal_intr(int signo, Sigfunc* func)
    {
        struct sigaction act, oact;
        act.sa_handler = func;
        sigemptyset(&act.sa_mask);
#ifdef SA_INTERRUPT
        act.sa_flags = SA_INTERRUPT;
#else
        act.sa_flags = SA_RESTART;
#endif
        if (sigaction(signo, &act, &oact) < 0)
            return SIG_ERR;
        return oact.sa_handler;
    }

    static void acceptAlarm(int signalVal)
    {
        signal_intr(SIGALRM, SIG_IGN);
        signal_intr(SIGALRM, acceptAlarm);
        alarm(GetTimeoutSeconds());
        FAIL_CHECK("ALARM: Timeout on test (signal " << signalVal << ")");
    }

    void SetUp()
    {
        signal_intr(SIGALRM, acceptAlarm);
        alarm(GetTimeoutSeconds());
    }

    void TearDown()
    {
        alarm(0);  // cancel alarm
    }
};

#endif /* UNITTEST_H_ */
