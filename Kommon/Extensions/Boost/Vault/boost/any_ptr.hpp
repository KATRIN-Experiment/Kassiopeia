// -----------------------------------------------------------------------------
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
//
// by Cassio Neri
// Complete article available at:
// http://drdobbs.com/cpp/229401004
// -----------------------------------------------------------------------------

#ifndef ANY_PTR_H_INCLUDED_
#define ANY_PTR_H_INCLUDED_

#include <algorithm>
#include <boost/shared_ptr.hpp>

namespace detail {
class paparazzo;
};

namespace boost {

class any_ptr;

// A dummy specialization of ::boost::shared_ptr<paparazzo> that exposes
// ::boost::shared_ptr<T>'s member px.
template <>
class shared_ptr< ::detail::paparazzo> {

    friend class any_ptr;

    // This class isn't meant to be constructed.
    shared_ptr();

    template <typename T>
    static void change_px(shared_ptr<T>& orig, void* ptr) {
        orig.px = static_cast<T*>(ptr);
    }

}; // class shared_ptr< ::detail::paparazzo>



//------------------------------------------------------------------------------
// A smart pointer to any type.
//
// This class wrapps up a ::boost::shared_ptr<void> and makes type annotation
// allowing for safe run-time casts.
//------------------------------------------------------------------------------
class any_ptr {

    ::boost::shared_ptr<void> ptr_;
    void (any_ptr::*thr_)() const;

    template <typename T>
    void thrower() const { throw static_cast<T*>(ptr_.get()); }

public:

    // Default constructor builds a NULL any_ptr.
    any_ptr() : thr_(0) {}

    // Constructor taking a T*.
    template <typename T>
    explicit any_ptr(T* ptr) : ptr_(ptr), thr_(&any_ptr::thrower<T>) {}

    // Constructor taking a ::boost::shared_ptr<T>.
    template <typename T>
    explicit any_ptr(::boost::shared_ptr<T> ptr) : ptr_(ptr),
    thr_(&any_ptr::thrower<T>) {}

    // Automatic conversion to ::boost::shared_ptr<T>.
    template <typename U>
    operator ::boost::shared_ptr<U>() const {

        if (ptr_) {

            try { (this->*thr_)(); }

            catch (U* p) {
                ::boost::shared_ptr<U> result = ::boost::static_pointer_cast<U>(ptr_);
                ::boost::shared_ptr< ::detail::paparazzo>::change_px(result, p);
                return result;
            }

            catch (...) {}
        }

        return ::boost::shared_ptr<U>();
    }

    // Automatic conversion to bool.
    operator bool() const { return ptr_; }

    // Non throwing swap.
    void swap(any_ptr& other) /* throw() */ {
        ptr_.swap(other.ptr_);
        ::std::swap(thr_, other.thr_);
    }

}; // class any_ptr

} // namespace boost

namespace std {

// Specialization of std::swap.
template <>
void swap<boost::any_ptr>(boost::any_ptr& a, boost::any_ptr& b) {
    a.swap(b);
}

} // namespace std

#endif // ANY_PTR_H_INCLUDED
