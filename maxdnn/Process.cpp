#define _XOPEN_SOURCE 500
#include <maxdnn/Process.hpp>
#include <unistd.h>

namespace maxdnn
{
    namespace Process
    {
        void sleep(double seconds)
        {
            if (seconds < 0.0) {
                seconds = 0.0;
            }
            ::usleep(static_cast<unsigned int>(seconds*1.0e6));
        }
    }
}
