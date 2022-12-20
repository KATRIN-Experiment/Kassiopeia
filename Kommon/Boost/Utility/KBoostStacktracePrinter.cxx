#include <KMessage.h>

#include <boost/stacktrace.hpp>
#include <ostream>

using namespace katrin;

void BoostStacktracePrinter(std::ostream& aStream)
{
    aStream << "stack trace:" << std::endl;
    for (auto& frame : boost::stacktrace::stacktrace()) {
        boost::stacktrace::detail::to_string_impl impl;
        aStream << impl(frame.address()) << " [" << frame.address() << "]" << std::endl;
    }
}

auto defaultStacktracePrinter = KMessageTable::GetInstance().SetStacktracePrinterCallback(BoostStacktracePrinter);
