#include "KStringUtils.h"
#include <boost/filesystem.hpp>

using namespace katrin;

std::string KStringUtils::GroupDigits(std::string str, const std::string& sep)
{
    boost::trim(str);
    size_t i = str.find_first_not_of("-+0123456789");

    if (i == std::string::npos)
        i = str.length();
    if (i < 4)
        return str;

    for (i -= 3; i > 0 && i < str.length(); i -= 3)
        str.insert(i, sep);

    return str;
}

std::string KStringUtils::RandomAlphaNum(size_t length)
{
    static const std::string alphanums = "0123456789"
                                         "abcdefghijklmnopqrstuvwxyz"
                                         "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    std::string result;
    result.reserve(length);
    while (length--)
        result += alphanums[KRandom::GetInstance().Uniform<size_t>(0, alphanums.length() - 1)];

    return result;
}

void KStringUtils::HexDump(const void *data, size_t length, std::ostream& os)
{
    // Code adapted from https://gist.github.com/domnikl/af00cc154e3da1c5d965

    const unsigned nbytes = 16;  // bytes displayed per line of output
    const unsigned addrw = fmax(4, log(length)/log(16));  // max. digits of displayed offset

    size_t i;
    unsigned char linebuf[nbytes+1];
    unsigned char *pc = (unsigned char*)data;

    // Process every byte in the data.
    for (i = 0; i < length; i++) {
        // Multiple of nbytes means new line (with line offset).

        if ((i % nbytes) == 0) {
            // Just don't print ASCII for the zeroth line.
            if (i != 0)
                os << "  " << linebuf << "\n";

            // Output the offset.
            os << "  " << std::noshowbase << std::setfill('0') << std::setw(addrw) << std::hex << i;
        }

        // Now the hex code for the specific character.
        os << " " << std::noshowbase << std::setfill('0') << std::setw(2) << std::hex << (unsigned short)pc[i];

        // And store a printable ASCII character for later.
        if ((pc[i] < 0x20) || (pc[i] > 0x7e)) {
            linebuf[i % nbytes] = '.';
        } else {
            linebuf[i % nbytes] = pc[i];
        }

        linebuf[(i % nbytes) + 1] = '\0';
    }

    // Pad out last line if not exactly 16 characters.
    while ((i % nbytes) != 0) {
        os << "   ";
        i++;
    }

    // And print the final ASCII bit.
    os << "  " << linebuf << "\n";
}
