#ifndef KSTATICASSERT_H_
#define KSTATICASSERT_H_

namespace katrin
{

    template< bool XAssertion >
    class KAssertion
    {
        public:
            enum{ Value = -1 };
    };

    template< >
    class KAssertion< true >
    {
        public:
            enum{ Value = 1 };
    };

}

#define KSTATICASSERT(anAssertion, aMessage) typedef bool ASSERTION_FAILED_##aMessage [ ::katrin::KAssertion< ((anAssertion) == 0 ? false : true) >::Value ];

#endif
