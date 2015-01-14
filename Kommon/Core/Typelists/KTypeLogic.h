#ifndef KTYPECOMPARISON_H_
#define KTYPECOMPARISON_H_

namespace katrin
{

    template< class XLeft, class XRight >
    class KTypeEqual
    {
        public:
            enum
            {
                Value = 0
            };
    };

    template< class XMatch >
    class KTypeEqual< XMatch, XMatch >
    {
        public:
            enum
            {
                Value = 1
            };
    };

    template< int XCondition, class XConditionMet, class XConditionNotMet >
    class KTypeIf;

    template< class XConditionMet, class XConditionNotMet >
    class KTypeIf< 1, XConditionMet, XConditionNotMet >
    {
        public:
            typedef XConditionMet Type;
    };

    template< class XConditionMet, class XConditionNotMet >
    class KTypeIf< 0, XConditionMet, XConditionNotMet >
    {
        public:
            typedef XConditionNotMet Type;
    };
}

#endif
