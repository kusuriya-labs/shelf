#include <shelf/runtime/tp/tp_transports.h>

#include <cstdint>
#include <cstdio>

using shelf::runtime::TpPeerLinkMap;
using shelf::runtime::tp_peer_link_map_all;
using shelf::runtime::tp_peer_link_map_disabled;
using shelf::runtime::tp_peer_link_map_none;

static int g_fail = 0;

static void check(bool cond, const char* msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg);
    }
}

int main() {
    const TpPeerLinkMap all{
        {0u, 1u, 1u},
        {1u, 0u, 1u},
        {1u, 1u, 0u},
    };
    check(tp_peer_link_map_all(all), "all directions usable");
    check(!tp_peer_link_map_none(all), "not all directions disabled");
    check(tp_peer_link_map_disabled(all) == 0u, "diagonal is excluded from selection");

    TpPeerLinkMap partial = all;
    partial[1][0] = 0u;
    check(!tp_peer_link_map_all(partial), "one disabled direction breaks the all predicate");
    check(!tp_peer_link_map_none(partial), "one disabled direction is not a full fallback");
    check(tp_peer_link_map_disabled(partial) == 1u, "disabled direction is counted");

    TpPeerLinkMap none(3, std::vector<std::uint8_t>(3, 0u));
    check(tp_peer_link_map_none(none), "all directions disabled");
    check(!tp_peer_link_map_all(none), "all disabled is not all usable");
    check(tp_peer_link_map_disabled(none) == 6u, "all entries disabled");

    check(tp_peer_link_map_all({}) && tp_peer_link_map_none({}),
          "an empty map is vacuously all and none");
    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
