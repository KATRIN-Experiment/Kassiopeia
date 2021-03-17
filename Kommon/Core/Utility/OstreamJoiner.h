#ifndef KOMMON_CORE_OSTREAM_JOINER_H
#define KOMMON_CORE_OSTREAM_JOINER_H


#include <iosfwd>
#include <iterator>

namespace katrin
{

namespace Kommon
{

/// Output iterator that inserts a delimiter between elements.
template<typename Delimiter, typename Character = char, typename Traits = std::char_traits<Character>>
class OstreamJoiner
{
  public:
    typedef Character char_type;
    using traits_type = Traits;
    using ostream_type = std::basic_ostream<Character, Traits>;
    using iterator_category = std::output_iterator_tag;
    using value_type = void;
    using difference_type = void;
    using pointer = void;
    using reference = void;

    OstreamJoiner(ostream_type& os, const Delimiter& delimiter) : output(os), delimiter(delimiter) {}

    OstreamJoiner(ostream_type& os, Delimiter&& delimiter) : output(os), delimiter(std::move(delimiter)) {}

    template<typename Value> OstreamJoiner& operator=(const Value& value)
    {
        if (!firstItem)
            output << delimiter;
        firstItem = false;
        output << value;
        return *this;
    }

    OstreamJoiner& operator*() noexcept
    {
        return *this;
    }
    OstreamJoiner& operator++() noexcept
    {
        return *this;
    }
    OstreamJoiner& operator++(int) noexcept
    {
        return *this;
    }

  private:
    ostream_type& output;
    Delimiter delimiter;
    bool firstItem = true;
};

/// Object generator for OstreamJoiner.
template<typename Character, typename Traits, typename Delimiter>
inline OstreamJoiner<std::decay_t<Delimiter>, Character, Traits>
MakeOstreamJoiner(std::basic_ostream<Character, Traits>& os, Delimiter&& delimiter)
{
    return {os, std::forward<Delimiter>(delimiter)};
}

} /* namespace Kommon */

}  // namespace katrin

#endif
