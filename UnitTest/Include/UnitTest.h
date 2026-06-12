#ifndef UNITTEST_H_
#define UNITTEST_H_

#include "doctest/doctest.h"
#include <csignal>
#include <cmath>
#include <unistd.h>

namespace kasper_unittest
{
template<class Fixture>
class FixtureWrapper : public Fixture
{
  public:
    FixtureWrapper()
    {
        this->SetUp();
    }
    ~FixtureWrapper()
    {
        this->TearDown();
    }
};
}  // namespace kasper_unittest

#define TEST(test_suite_name, test_name) TEST_CASE(#test_suite_name "." #test_name)
#define TEST_F(fixture_name, test_name)                                                                  \
    TEST_CASE_FIXTURE(kasper_unittest::FixtureWrapper<fixture_name>, #fixture_name "." #test_name)

#define EXPECT_EQ(a, b) CHECK_EQ(a, b)
#define ASSERT_EQ(a, b) REQUIRE_EQ(a, b)
#define EXPECT_NE(a, b) CHECK_NE(a, b)
#define ASSERT_NE(a, b) REQUIRE_NE(a, b)
#define EXPECT_LT(a, b) CHECK_LT(a, b)
#define ASSERT_LT(a, b) REQUIRE_LT(a, b)
#define EXPECT_LE(a, b) CHECK_LE(a, b)
#define ASSERT_LE(a, b) REQUIRE_LE(a, b)
#define EXPECT_GT(a, b) CHECK_GT(a, b)
#define ASSERT_GT(a, b) REQUIRE_GT(a, b)
#define EXPECT_GE(a, b) CHECK_GE(a, b)
#define ASSERT_GE(a, b) REQUIRE_GE(a, b)
#define EXPECT_TRUE(a) CHECK(a)
#define ASSERT_TRUE(a) REQUIRE(a)
#define EXPECT_FALSE(a) CHECK_FALSE(a)
#define ASSERT_FALSE(a) REQUIRE_FALSE(a)
#define EXPECT_NEAR(a, b, eps) CHECK(std::fabs((a) - (b)) <= (eps))
#define ASSERT_NEAR(a, b, eps) REQUIRE(std::fabs((a) - (b)) <= (eps))
#define EXPECT_DOUBLE_EQ(a, b) CHECK_EQ(a, b)
#define ASSERT_DOUBLE_EQ(a, b) REQUIRE_EQ(a, b)
#define EXPECT_DOUBLE_LT(a, b) CHECK_LT(a, b)
#define ASSERT_DOUBLE_LT(a, b) REQUIRE_LT(a, b)
#define EXPECT_THROW(statement, exception_type) CHECK_THROWS_AS(statement, exception_type)
#define ASSERT_THROW(statement, exception_type) REQUIRE_THROWS_AS(statement, exception_type)
#define ASSERT_ANY_THROW(statement) REQUIRE_THROWS(statement)

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

#define EXPECT_PTR(p) EXPECT_TRUE((p) != nullptr)
#define ASSERT_PTR(p) ASSERT_TRUE((p) != nullptr)

#define EXPECT_NULL(p) EXPECT_TRUE((p) == nullptr)
#define ASSERT_NULL(p) ASSERT_TRUE((p) == nullptr)

#define EXPECT_STRING_EQ(a, b) EXPECT_EQ(std::string(a), std::string(b))
#define ASSERT_STRING_EQ(a, b) ASSERT_EQ(std::string(a), std::string(b))

#define EXPECT_VECTOR_NULL(a) EXPECT_DOUBLE_LT((a).Magnitude(), ROUND_ERROR_DOUBLE)
#define ASSERT_VECTOR_NULL(a) ASSERT_DOUBLE_LT((a).Magnitude(), ROUND_ERROR_DOUBLE)

#define EXPECT_VECTOR_NEAR(a, b) EXPECT_LT(((a) - (b)).Magnitude(), ROUND_ERROR_DOUBLE)
#define ASSERT_VECTOR_NEAR(a, b) ASSERT_LT(((a) - (b)).Magnitude(), ROUND_ERROR_DOUBLE)


/**
 * A fancy fixture class which adds timeouts to fixtures/tests deriving from this one.
 * @author J. Behrens
 */
class TimeoutTest
{
  public:
    TimeoutTest() = default;
    virtual ~TimeoutTest() = default;

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
        FAIL("ALARM: Timeout on test (signal " << signalVal << ")");
    }

    virtual void SetUp()
    {
        signal_intr(SIGALRM, acceptAlarm);
        alarm(GetTimeoutSeconds());
    }

    virtual void TearDown()
    {
        alarm(0);  // cancel alarm
    }
};

#endif /* UNITTEST_H_ */
