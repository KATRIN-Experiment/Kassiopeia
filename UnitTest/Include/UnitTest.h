#ifndef UNITTEST_H_
#define UNITTEST_H_

#include "gtest/gtest.h"
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

#define EXPECT_PTR(p) EXPECT_FALSE((p) == NULL)
#define ASSERT_PTR(p) ASSERT_FALSE((p) == NULL)

#define EXPECT_NULL(p) EXPECT_TRUE((p) == NULL)
#define ASSERT_NULL(p) ASSERT_TRUE((p) == NULL)

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
class TimeoutTest : public ::testing::Test
{
  public:
    static int GetTimeoutSeconds()
    {
        return 30;
    }  // all tests should complete within 30 seconds

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
        const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        FAIL() << "ALARM: Timeout on test " << test_info->name() << " (signal " << signalVal << ")";
    }

    void SetUp() override
    {
        signal_intr(SIGALRM, acceptAlarm);
        alarm(GetTimeoutSeconds());
    }

    void TearDown() override
    {
        alarm(0);  // cancel alarm
    }
};

#endif /* UNITTEST_H_ */
