#include <shelf/models/qwen35/state/state_types.h>

#include <cassert>
#include <limits>
#include <cstdio>

using namespace ps::qwen35;

int main() {
    const PageId page = 42;
    const SequenceSlotId slot = 7;

    assert(page == 42);
    assert(slot == 7);

    assert(kInvalidPageId == std::numeric_limits<PageId>::max());
    assert(kInvalidSlot == std::numeric_limits<SequenceSlotId>::max());

    assert(kInvalidPageId != 0);
    assert(kInvalidSlot != 0);

    printf("Paged Types sanity test passed\n");
    return 0;
}
