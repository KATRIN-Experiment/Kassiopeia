#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

namespace
{
constexpr const char* kRunStatus = "[ RUN       ] ";
constexpr const char* kOkStatus = "[ OK        ] ";
constexpr const char* kFailedStatus = "[ FAILED    ] ";
constexpr const char* kSkippedStatus = "[ SKIPPED   ] ";

class TestCaseProgressListener : public doctest::IReporter
{
  public:
    explicit TestCaseProgressListener(const doctest::ContextOptions& options) : fOptions(options) {}

    void report_query(const doctest::QueryData&) override {}
    void test_run_start() override {}
    void test_run_end(const doctest::TestRunStats&) override {}

    void test_case_start(const doctest::TestCaseData& in) override
    {
        if (!Enabled() || in.m_no_output)
            return;

        fCurrentTest = in.m_name;
        (*fOptions.cout) << kRunStatus << fCurrentTest << '\n';
    }

    void test_case_reenter(const doctest::TestCaseData&) override {}

    void test_case_end(const doctest::CurrentTestCaseStats& in) override
    {
        if (!Enabled() || fCurrentTest.empty())
            return;

        (*fOptions.cout) << (in.testCaseSuccess ? kOkStatus : kFailedStatus) << fCurrentTest << '\n';
        fCurrentTest.clear();
    }

    void test_case_exception(const doctest::TestCaseException&) override {}
    void subcase_start(const doctest::SubcaseSignature&) override {}
    void subcase_end() override {}
    void log_assert(const doctest::AssertData&) override {}
    void log_message(const doctest::MessageData&) override {}

    void test_case_skipped(const doctest::TestCaseData& in) override
    {
        if (!Enabled() || in.m_no_output)
            return;

        (*fOptions.cout) << kSkippedStatus << in.m_name << '\n';
    }

  private:
    bool Enabled() const
    {
        return (fOptions.cout != nullptr) && !fOptions.quiet;
    }

    const doctest::ContextOptions& fOptions;
    std::string fCurrentTest;
};
}  // namespace

REGISTER_LISTENER("test-progress-listener", 1, TestCaseProgressListener);

int main(int argc, char** argv)
{
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    return context.run();
}
